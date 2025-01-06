#import <metal_stdlib>
using namespace metal;

#define FXAA_REDUCE_MIN (1.0/128.0)
#define FXAA_REDUCE_MUL (1.0/8.0)
#define FXAA_SPAN_MAX 8.0

#define FXAA_DISCARD

constexpr sampler linear(filter::linear,coord::normalized,address::mirrored_repeat);

kernel void processimage(
    texture2d<float,access::sample> src[[texture(0)]],
    texture2d<float,access::write> dst[[texture(1)]],
    constant float2 &resolution[[buffer(0)]],
    uint2 gid[[thread_position_in_grid]]) {
        
        simd::float3 luma = float3(0.299,0.587,0.114);
        
        simd::float2 uv = (float2(gid)+0.5)/resolution;
        simd::float2 res = 1.0/(resolution-1.0);
        
        float3 rgbM = src.sample(linear,uv).xyz;
        float lumaM = dot(rgbM,luma);
        
        float fxaaQualityEdgeThreshold = 0.0833;
        float fxaaQualityEdgeThresholdMin = 0.05;
        
        float lumaS = dot(luma,src.sample(linear,uv+float2( 0, 1)*res).xyz);
        float lumaE = dot(luma,src.sample(linear,uv+float2( 1, 0)*res).xyz);
        float lumaN = dot(luma,src.sample(linear,uv+float2( 0,-1)*res).xyz);
        float lumaW = dot(luma,src.sample(linear,uv+float2(-1, 0)*res).xyz);
        
        float maxSM = max(lumaS,lumaM);
        float minSM = min(lumaS,lumaM);
        float maxESM = max(lumaE,maxSM);
        float minESM = min(lumaE,minSM);
        float maxWN = max(lumaN,lumaW);
        float minWN = min(lumaN,lumaW);
        float rangeMax = max(maxWN,maxESM);
        float rangeMin = min(minWN,minESM);
        float rangeMaxScaled = rangeMax*fxaaQualityEdgeThreshold;
        float rangeMaxClamped = max(fxaaQualityEdgeThresholdMin, rangeMaxScaled);
        
        if((rangeMax-rangeMin)<rangeMaxClamped) {
        
#ifdef FXAA_DISCARD
            dst.write(float4(0.0),gid);
#else
            dst.write(float4(rgbM,1.0),gid);
#endif
            
        }
        else {
            
            float lumaNW = dot(luma,src.sample(linear,uv+float2(-1,-1)*res).xyz);
            float lumaNE = dot(luma,src.sample(linear,uv+float2( 1,-1)*res).xyz);
            float lumaSW = dot(luma,src.sample(linear,uv+float2(-1, 1)*res).xyz);
            float lumaSE = dot(luma,src.sample(linear,uv+float2( 1, 1)*res).xyz);
            
            float lumaMin = min(lumaM,min(min(lumaNW,lumaNE),min(lumaSW,lumaSE)));
            float lumaMax = max(lumaM,max(max(lumaNW,lumaNE),max(lumaSW,lumaSE)));
            
            float2 dir = {
                -((lumaNW+lumaNE)-(lumaSW+lumaSE)),
                 ((lumaNW+lumaSW)-(lumaNE+lumaSE))
            };
            
            float dirReduce = max((lumaNW+lumaNE+lumaSW+lumaSE)*(0.25*FXAA_REDUCE_MUL),FXAA_REDUCE_MIN);
            
            float rcpDirMin = 1.0/(min(abs(dir.x),abs(dir.y))+dirReduce);
            dir = min(float2(FXAA_SPAN_MAX,FXAA_SPAN_MAX),max(float2(-FXAA_SPAN_MAX,-FXAA_SPAN_MAX),dir*rcpDirMin))*res;
            
            float4 rgbA = 0.5*(src.sample(linear,uv+dir*(1.0/3.0-0.5))+src.sample(linear,uv+dir*(2.0/3.0-0.5)));
            float4 rgbB = 0.5*(rgbA+0.5*(src.sample(linear,uv+dir*(0.0/3.0-0.5))+src.sample(linear,uv+dir*(3.0/3.0-0.5))));
            
            float lumaB = dot(rgbB.xyz,luma);
            dst.write(((lumaB<lumaMin)||(lumaB>lumaMax))?rgbA:rgbB,gid);
        }
}