#pragma once
#include "IUnityInterface.h"

//#ifndef __OBJC__
//    #error metal plugin is objc code.
//#endif
#ifndef __clang__
    #error only clang compiler is supported.
#endif
namespace NS {
    class Bundle;
}
namespace MTL {
    class Device;
    class CommandBuffer;
    class CommandEncoder;
    class Texture;
    class RenderPassDescriptor;
}

// Should only be used on the rendering thread unless noted otherwise.
UNITY_DECLARE_INTERFACE(IUnityGraphicsMetal)
{
    NS::Bundle*               (UNITY_INTERFACE_API * MetalBundle)();
    MTL::Device* (UNITY_INTERFACE_API * MetalDevice)();

    MTL::CommandBuffer* (UNITY_INTERFACE_API * CurrentCommandBuffer)();

    // for custom rendering support there are two scenarios:
    // you want to use current in-flight MTLCommandEncoder (NB: it might be nil)
    MTL::CommandEncoder* (UNITY_INTERFACE_API * CurrentCommandEncoder)();
    // or you might want to create your own encoder.
    // In that case you should end unity's encoder before creating your own and end yours before returning control to unity
    void(UNITY_INTERFACE_API * EndCurrentCommandEncoder)();

    // returns MTLRenderPassDescriptor used to create current MTLCommandEncoder
    MTL::RenderPassDescriptor* (UNITY_INTERFACE_API * CurrentRenderPassDescriptor)();

    // converting trampoline UnityRenderBufferHandle into native RenderBuffer
    UnityRenderBuffer(UNITY_INTERFACE_API * RenderBufferFromHandle)(void* bufferHandle);

    // access to RenderBuffer's texure
    // NB: you pass here *native* RenderBuffer, acquired by calling (C#) RenderBuffer.GetNativeRenderBufferPtr
    // AAResolvedTextureFromRenderBuffer will return nil in case of non-AA RenderBuffer or if called for depth RenderBuffer
    // StencilTextureFromRenderBuffer will return nil in case of no-stencil RenderBuffer or if called for color RenderBuffer
    MTL::Texture* (UNITY_INTERFACE_API * TextureFromRenderBuffer)(UnityRenderBuffer buffer);
    MTL::Texture* (UNITY_INTERFACE_API * AAResolvedTextureFromRenderBuffer)(UnityRenderBuffer buffer);
    MTL::Texture* (UNITY_INTERFACE_API * StencilTextureFromRenderBuffer)(UnityRenderBuffer buffer);
};
UNITY_REGISTER_INTERFACE_GUID(0x992C8EAEA95811E5ULL, 0x9A62C4B5B9876117ULL, IUnityGraphicsMetal)
