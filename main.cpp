/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


//--------------------------------------------------------------------------------------------------
// Very simple Vulkan example which render an image and save it to disk
// without creating any window context.
//

#include <vulkan/vulkan.hpp>

#include "nvh/fileoperations.hpp"
#include "nvh/inputparser.h"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/vulkanhppsupport.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Allocator : using dedicated

#include "nvvk/resourceallocator_vk.hpp"

#include "nvpsystem.hpp"
#include <iostream>
#include <random>

// Globals
static int const                SAMPLE_SIZE_WIDTH  = 800;
static int const                SAMPLE_SIZE_HEIGHT = 600;
static std::vector<std::string> s_defaultSearchPaths;
static float                    s_animTime{0.0};

//--------------------------------------------------------------------------------------------------
// Default example base class
//
class DummyExample
{
public:
  void setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex)
  {
    m_instance           = instance;
    m_device             = device;
    m_physicalDevice     = physicalDevice;
    m_graphicsQueueIndex = graphicsQueueIndex;

    m_alloc.init(m_device, m_physicalDevice);
  }

  //--------------------------------------------------------------------------------------------------
  // Rendering the scene to a frame buffer
  void offlineRender()
  {
    std::array<vk::ClearValue, 2> clearValues;
    clearValues[0].color = std::array<float, 4>({0.1f, 0.1f, 0.4f, 0.f});
    clearValues[1].setDepthStencil({1.0f, 0});

    // Command Buffer Pool Generator
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);

    // Preparing the rendering
    vk::CommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();

    // Rendering pass to framebuffer
    vk::RenderPassBeginInfo renderPassBeginInfo;
    renderPassBeginInfo.setRenderPass(m_renderPass);
    renderPassBeginInfo.setFramebuffer(m_framebuffer);
    renderPassBeginInfo.setRenderArea({{}, m_size});
    renderPassBeginInfo.setClearValueCount(2);
    renderPassBeginInfo.setPClearValues(clearValues.data());
    cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

    // Viewport and scissor
    cmdBuf.setViewport(0, {vk::Viewport(0.0f, 0.0f, static_cast<float>(m_size.width), static_cast<float>(m_size.height), 0.0f, 1.0f)});
    cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

    // Rendering the full-screen pixel shader
    auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
    cmdBuf.pushConstants<float>(m_pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, s_animTime);
    cmdBuf.pushConstants<float>(m_pipelineLayout, vk::ShaderStageFlagBits::eFragment, sizeof(float), aspectRatio);
    cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);
    cmdBuf.draw(3, 1, 0, 0);  // No vertices, it is implicitly done in the veetex shader

    // Done and submit execution
    cmdBuf.endRenderPass();
    genCmdBuf.submitAndWait(cmdBuf);
  }

  //--------------------------------------------------------------------------------------------------
  // Copy the image to a buffer - this linearize the image memory
  //
  void imageToBuffer(const nvvk::Texture& imgIn, const vk::Buffer& pixelBufferOut)
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    vk::CommandBuffer cmdBuff = genCmdBuf.createCommandBuffer();

    // Make the image layout eTransferSrcOptimal to copy to buffer
    vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    nvvkpp::cmdBarrierImageLayout(cmdBuff, imgIn.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal, subresourceRange);

    // Copy the image to the buffer
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setImageExtent(vk::Extent3D(m_size, 1));
    cmdBuff.copyImageToBuffer(imgIn.image, vk::ImageLayout::eTransferSrcOptimal, pixelBufferOut, {copyRegion});

    // Put back the image as it was
    nvvkpp::cmdBarrierImageLayout(cmdBuff, imgIn.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral, subresourceRange);
    genCmdBuf.submitAndWait(cmdBuff);
  }

  //--------------------------------------------------------------------------------------------------
  // Save the image to disk
  //
  void saveImage(const std::string& outFilename)
  {
    // Create a temporary buffer to hold the pixels of the image
    vk::BufferUsageFlags usage{vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst};
    vk::DeviceSize       bufferSize = 4 * sizeof(uint8_t) * m_size.width * m_size.height;
    nvvk::Buffer pixelBuffer        = m_alloc.createBuffer(bufferSize, usage, vk::MemoryPropertyFlagBits::eHostVisible);

    imageToBuffer(m_colorTexture, pixelBuffer.buffer);

    // Write the buffer to disk
    const void* data = m_alloc.map(pixelBuffer);
    stbi_write_png(outFilename.c_str(), m_size.width, m_size.height, 4, data, 0);
    m_alloc.unmap(pixelBuffer);

    // Destroy temporary buffer
    m_alloc.destroy(pixelBuffer);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline of this example
  //
  void createPipeline()
  {
    const auto& paths = s_defaultSearchPaths;

    // Pipeline Layout: The layout of the shader needs only Push Constants: we are using parameters, time and aspect ratio
    vk::PushConstantRange        push_constants = {vk::ShaderStageFlagBits::eFragment, 0, 2 * sizeof(float)};
    vk::PipelineLayoutCreateInfo layout_info;
    layout_info.setPushConstantRangeCount(1);
    layout_info.setPPushConstantRanges(&push_constants);
    m_pipelineLayout = m_device.createPipelineLayout(layout_info);

    // Pipeline: completely generic, no vertices
    nvvkpp::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_pipelineLayout, m_renderPass);
    pipelineGenerator.addShader(nvh::loadFile("shaders/vert_shader.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
    pipelineGenerator.addShader(nvh::loadFile("shaders/frag_shader.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
    pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
    m_pipeline = pipelineGenerator.createPipeline();
  }

  //--------------------------------------------------------------------------------------------------
  // Creating an offscreen frame buffer and the associated render pass
  //
  void createFramebuffer(const VkExtent2D& size)
  {
    m_size = size;

    // Creating the color image
    auto colorCreateInfo = nvvkpp::makeImage2DCreateInfo(size, m_colorFormat,
                                                       vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
                                                           | vk::ImageUsageFlagBits::eStorage);

    vk::SamplerCreateInfo samplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear};

    nvvk::Image             image         = m_alloc.createImage(colorCreateInfo);
    vk::ImageViewCreateInfo ivInfo        = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_colorTexture                        = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);
    m_colorTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Creating the depth buffer (not needed, but useful if the sample has to change)
    auto depthCreateInfo = nvvkpp::makeImage2DCreateInfo(size, m_depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment);
    nvvk::Image dImage = m_alloc.createImage(depthCreateInfo);

    vk::ImageViewCreateInfo depthStencilView;
    depthStencilView.setViewType(vk::ImageViewType::e2D);
    depthStencilView.setFormat(m_depthFormat);
    depthStencilView.setSubresourceRange({vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
    depthStencilView.setImage(dImage.image);
    m_depthTexture = m_alloc.createTexture(dImage, depthStencilView);

    // Setting the image layout for both color and depth
    {
      nvvkpp::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
      vk::CommandBuffer    cmdBuf = genCmdBuf.createCommandBuffer();
      nvvkpp::cmdBarrierImageLayout(cmdBuf, m_colorTexture.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
      nvvkpp::cmdBarrierImageLayout(cmdBuf, m_depthTexture.image, vk::ImageLayout::eUndefined,
                                     vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageAspectFlagBits::eDepth);

      genCmdBuf.submitAndWait({cmdBuf});
    }

    // Creating a renderpass for the offscreen
    if(!m_renderPass)
    {
      m_renderPass = nvvkpp::createRenderPass(m_device, {m_colorFormat}, m_depthFormat, 1, true, true,
                                            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
    }

    // Creating the frame buffer for offscreen
    std::vector<vk::ImageView> attachments = {m_colorTexture.descriptor.imageView, m_depthTexture.descriptor.imageView};

    m_device.destroy(m_framebuffer);
    vk::FramebufferCreateInfo info;
    info.setRenderPass(m_renderPass);
    info.setAttachmentCount(2);
    info.setPAttachments(attachments.data());
    info.setWidth(size.width);
    info.setHeight(size.height);
    info.setLayers(1);
    m_framebuffer = m_device.createFramebuffer(info);
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  void destroy()  // override
  {
    m_alloc.destroy(m_colorTexture);
    m_alloc.destroy(m_depthTexture);

    m_device.waitIdle();
    m_device.destroy(m_pipeline);
    m_device.destroy(m_pipelineLayout);
    m_device.destroy(m_framebuffer);
    m_device.destroy(m_renderPass);
  }


private:
  vk::Instance                     m_instance;        // Vulkan instance
  vk::Device                       m_device;          // Logical GPU
  vk::PhysicalDevice               m_physicalDevice;  // Physical GPU
  vk::Pipeline                     m_pipeline;        // Graphic pipeline
  vk::PipelineLayout               m_pipelineLayout;  // Graphic pipeline layout
  nvvkpp::ResourceAllocatorDedicated m_alloc;           // Allocator for buffer, images
  nvvk::Texture                    m_colorTexture;    // colored image
  nvvk::Texture                    m_depthTexture;    // depth buffer
  vk::Framebuffer                  m_framebuffer;     // color + depth framebuffer
  vk::RenderPass                   m_renderPass;      // Base render pass
  vk::Extent2D                     m_size{0, 0};      // Size of the window
  uint32_t                         m_graphicsQueueIndex{VK_QUEUE_FAMILY_IGNORED};
  vk::Format                       m_colorFormat{vk::Format::eR8G8B8A8Unorm};
  vk::Format                       m_depthFormat{vk::Format::eD32Sfloat};
};

//--------------------------------------------------------------------------------------------------
// Entry of the example, see OPTIONS for the arguments
//
int main(int argc, char** argv)
{
  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  InputParser parser(argc, argv);
  if(parser.exist("-h"))
  {
    LOGE("\n offline.exe OPTIONS\n");
    LOGE("     -t : (float)   time   (default: 0.0). If time is < 0, then it will be random. \n");
    LOGE("     -s : (int int) size   (default: 800 600) \n");
    LOGE("     -o : (string)  output (default: \"result.png\") \n");
    exit(1);
  }
  s_animTime      = parser.getFloat("-t", 0.0);
  auto winSize    = parser.getInt2("-s", {SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT});
  auto outputFile = parser.getString("-o", "result.png");

  VkExtent2D sampleSize = {uint32_t(winSize[0]), uint32_t(winSize[1])};

  if(s_animTime < 0.0)
  {
    std::random_device                    rd;
    std::default_random_engine            e2(rd());
    std::uniform_real_distribution<float> dist(0, 1000);
    s_animTime = dist(e2);
  }

  s_defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      PROJECT_NAME,
  };

  LOGI("Starting Application\n");

  // Creating the Vulkan instance and device, with only defaults, no extension
  nvvk::Context vkctx;
  nvvk::ContextCreateInfo vkctxInfo{};
  vkctx.init(vkctxInfo);

  // Initialize Vulkan function pointers
  vk::DynamicLoader dl;
  auto              vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkctx.m_instance);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkctx.m_device);

  // Printing which GPU we are using
  vk::PhysicalDevice pd(vkctx.m_physicalDevice);
  LOGI("Using GPU: %s\n", pd.getProperties().deviceName.data());
  LOGI("Rendering:  time(%f), resolution(%d, %d)\n", s_animTime, sampleSize.width, sampleSize.height);

  // Running our example
  DummyExample example;
  example.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  example.createFramebuffer(sampleSize);                            // Framebuffer where it will render
  example.createPipeline();                                         // How the quad will be rendered: shaders and more
  example.offlineRender();                                          // Rendering

  LOGI("Saving Image: %s\n", outputFile.c_str());
  example.saveImage(outputFile);
  example.destroy();

  vkctx.deinit();
}
