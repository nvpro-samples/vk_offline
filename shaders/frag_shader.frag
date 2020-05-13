#version 450
layout (location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;


layout(push_constant) uniform shaderInformation
{
 float iTime;  
 float aspectRatio;
}
pushc;

void main()
{
  vec2 uv = outUV;

    uv.x *= pushc.aspectRatio;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(pushc.iTime+uv.xyx+vec3(0,2,4));

    for(int i=0; i<2; ++i) {
    	uv -= 0.5;
    	uv *= 5.;
        
        float radius = .5 + (sin(pushc.iTime*.3)*.15) - cos(pushc.iTime)*float(i)*.1;
    	vec2 fuv = fract(uv)-.5;
    	float x = smoothstep(radius*.9, radius*1.1,length(fuv));
    	float y = 1.0 - smoothstep(radius*.3, radius*.7,length(fuv));
    
    	col -= vec3(x+y);
    }

    // Output to screen
    fragColor = vec4(col,1.0);
}
