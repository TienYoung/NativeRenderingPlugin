
#include "RenderAPI.h"
#include "PlatformBase.h"


// Metal implementation of RenderAPI.


#if SUPPORT_METAL

#include "Unity/IUnityGraphicsMetal.h"
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>


class RenderAPI_Metal : public RenderAPI
{
public:
	RenderAPI_Metal();
	virtual ~RenderAPI_Metal() { }

	virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces);

	virtual bool GetUsesReverseZ() { return true; }

	virtual void DrawSimpleTriangles(const float worldMatrix[16], int triangleCount, const void* verticesFloat3Byte4);
    
    virtual void DrawMesh(const float worldMatrix[16], void* positionBuffer, void* colorBuffer, int count);

	virtual void* BeginModifyTexture(void* textureHandle, int textureWidth, int textureHeight, int* outRowPitch);
	virtual void EndModifyTexture(void* textureHandle, int textureWidth, int textureHeight, int rowPitch, void* dataPtr);

	virtual void* BeginModifyVertexBuffer(void* bufferHandle, size_t* outBufferSize);
	virtual void EndModifyVertexBuffer(void* bufferHandle);

private:
	void CreateResources();

private:
	IUnityGraphicsMetal*	m_MetalGraphics;
    MTL::Buffer*            m_VertexBuffer;
    MTL::Buffer*            m_PositionBuffer;
	MTL::Buffer*			m_ColorBuffer;
	MTL::Buffer*			m_ConstantBuffer;

	MTL::DepthStencilState*   m_DepthStencil;
    MTL::RenderPipelineState* m_Pipeline;
	MTL::RenderPipelineState* m_MeshPipeline;
};


RenderAPI* CreateRenderAPI_Metal()
{
	return new RenderAPI_Metal();
}


const int kVertexSize = 12 + 4;

// Simple vertex & fragment shader source
static const char kShaderSource[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct AppData\n"
"{\n"
"    float4x4 worldMatrix;\n"
"};\n"
"struct Vertex\n"
"{\n"
"    float3 pos [[attribute(0)]];\n"
"    float4 color [[attribute(1)]];\n"
"};\n"
"struct VSOutput\n"
"{\n"
"    float4 pos [[position]];\n"
"    half4  color;\n"
"};\n"
"struct FSOutput\n"
"{\n"
"    half4 frag_data [[color(0)]];\n"
"};\n"
"vertex VSOutput vertexMain(Vertex input [[stage_in]], constant AppData& my_cb [[buffer(0)]])\n"
"{\n"
"    VSOutput out = { my_cb.worldMatrix * float4(input.pos.xyz, 1), (half4)input.color };\n"
"    return out;\n"
"}\n"
"fragment FSOutput fragmentMain(VSOutput input [[stage_in]])\n"
"{\n"
"    FSOutput out = { input.color };\n"
"    return out;\n"
"}\n";

static const char kMeshShaderSource[] = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexData
{
    float4 position [[position]];
    float4 color; 
};

struct PrimitiveData 
{
};

using triangle_mesh_t = metal::mesh<VertexData, PrimitiveData, 3, 1, metal::topology::triangle>;

struct AppData
{
    float4x4 worldMatrix;
};

struct FragmentData
{
    VertexData vert;
    PrimitiveData prim;
};

[[mesh]]
void meshMain(triangle_mesh_t outputMesh, constant AppData& my_cb [[buffer(0)]], constant float4* position [[buffer(1)]], constant float4* color [[buffer(2)]])
{
    outputMesh.set_vertex(0, VertexData{my_cb.worldMatrix * position[0], color[0]});
    outputMesh.set_index(0, 0);
    outputMesh.set_vertex(1, VertexData{my_cb.worldMatrix * position[1], color[1]});
    outputMesh.set_index(1, 1);
    outputMesh.set_vertex(2, VertexData{my_cb.worldMatrix * position[2], color[2]});
    outputMesh.set_index(2, 2);
    outputMesh.set_primitive(0, PrimitiveData{});
    outputMesh.set_primitive_count(1);
}

[[fragment]]
float4 fragmentMain(FragmentData input [[stage_in]])
{
    return input.vert.color;
};
)";


void RenderAPI_Metal::CreateResources()
{
	MTL::Device* metalDevice = m_MetalGraphics->MetalDevice();
	NS::Error* error = nullptr;

	// Create shaders
	NS::String* srcStr = NS::String::alloc()->init(kShaderSource, NS::ASCIIStringEncoding);
	MTL::Library* shaderLibrary = metalDevice->newLibrary(srcStr, nullptr, &error);
	if(error != nullptr)
	{
		NS::String* desc   = error->localizedDescription();
		NS::String* reason = error->localizedFailureReason();
		::fprintf(stderr, "%s\n%s\n\n", desc ? desc->utf8String() : "<unknown>", reason ? reason->utf8String() : "");
	}

	MTL::Function* vertexFunction = shaderLibrary->newFunction(MTLSTR("vertexMain"));
	MTL::Function* fragmentFunction = shaderLibrary->newFunction(MTLSTR("fragmentMain"));

    // Create mesh shader
    srcStr = NS::String::alloc()->init(kMeshShaderSource, NS::ASCIIStringEncoding);
    MTL::Library* meshShaderLibrary = metalDevice->newLibrary(srcStr, nullptr, &error);
    if(error != nullptr)
    {
        NS::String* desc   = error->localizedDescription();
        NS::String* reason = error->localizedFailureReason();
        ::fprintf(stderr, "Desc: %s\nReason: %s\n\n", desc ? desc->utf8String() : "<unknown>", reason ? reason->utf8String() : "");
    }
    MTL::Function* meshFunction = meshShaderLibrary->newFunction(MTLSTR("meshMain"));
    MTL::Function* meshFragmentFunction = meshShaderLibrary->newFunction(MTLSTR("fragmentMain"));

    ::printf("MeshFunction %p", meshFunction);
    ::printf("Fragment Function %p", meshFragmentFunction);
    
	// Vertex / Constant buffers

#	if UNITY_OSX
	MTL::ResourceOptions bufferOptions = MTL::ResourceCPUCacheModeDefaultCache | MTL::ResourceStorageModeManaged;
#	else
	MTL::ResourceOptions bufferOptions = MTL::ResourceOptionCPUCacheModeDefault;
#	endif

	m_VertexBuffer = metalDevice->newBuffer(1024, bufferOptions);
	m_VertexBuffer->setLabel(MTLSTR("PluginVB"));
    m_PositionBuffer = metalDevice->newBuffer(1024, bufferOptions);
    m_PositionBuffer->setLabel(MTLSTR("PluginPB"));
    m_ColorBuffer = metalDevice->newBuffer(1024, bufferOptions);
    m_ColorBuffer->setLabel(MTLSTR("PluginColorB"));
	m_ConstantBuffer = metalDevice->newBuffer(16*sizeof(float), bufferOptions);
    m_ConstantBuffer->setLabel(MTLSTR("PluginCB"));

	// Vertex layout
	MTL::VertexDescriptor* vertexDesc = MTL::VertexDescriptor::vertexDescriptor();
	vertexDesc->attributes()->object(0)->setFormat(MTL::VertexFormatFloat3);
	vertexDesc->attributes()->object(0)->setOffset(0);
	vertexDesc->attributes()->object(0)->setBufferIndex(1);
	vertexDesc->attributes()->object(1)->setFormat(MTL::VertexFormatUChar4Normalized);
	vertexDesc->attributes()->object(1)->setOffset(3*sizeof(float));
	vertexDesc->attributes()->object(1)->setBufferIndex(1);
	vertexDesc->layouts()->object(1)->setStride(kVertexSize);
	vertexDesc->layouts()->object(1)->setStepFunction(MTL::VertexStepFunctionPerVertex);
	vertexDesc->layouts()->object(1)->setStepRate(1);

	// Pipeline

	NS::SharedPtr<MTL::RenderPipelineDescriptor> pipeDesc = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
	// Let's assume we're rendering into BGRA8Unorm...
	pipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

	pipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float_Stencil8);
	pipeDesc->setStencilAttachmentPixelFormat(MTL::PixelFormatDepth32Float_Stencil8);

    pipeDesc->setSampleCount(1);
	pipeDesc->colorAttachments()->object(0)->setBlendingEnabled(true);

	pipeDesc->setVertexFunction(vertexFunction);
	pipeDesc->setFragmentFunction(fragmentFunction);
	pipeDesc->setVertexDescriptor(vertexDesc);

	m_Pipeline = metalDevice->newRenderPipelineState(pipeDesc.get(), &error);
	if (error != nullptr)
	{
		::fprintf(stderr, "Metal: Error creating pipeline state: %s\n%s\n", error->localizedDescription()->utf8String(), error->localizedFailureReason()->utf8String());
		error = nullptr;
	}

	// Depth/Stencil state
	NS::SharedPtr<MTL::DepthStencilDescriptor> depthDesc = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    depthDesc->setDepthCompareFunction(GetUsesReverseZ() ? MTL::CompareFunctionGreaterEqual : MTL::CompareFunctionLessEqual);
	depthDesc->setDepthWriteEnabled(false);
	m_DepthStencil = metalDevice->newDepthStencilState(depthDesc.get());
    
    // Mesh shader pipeline
    NS::SharedPtr<MTL::MeshRenderPipelineDescriptor> meshDesc = NS::TransferPtr(MTL::MeshRenderPipelineDescriptor::alloc()->init());
    meshDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    
    meshDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float_Stencil8);
    meshDesc->setStencilAttachmentPixelFormat(MTL::PixelFormatDepth32Float_Stencil8);
    
    meshDesc->setRasterSampleCount(1);
    meshDesc->colorAttachments()->object(0)->setBlendingEnabled(true);
    meshDesc->setMeshFunction(meshFunction);
    meshDesc->setFragmentFunction(meshFragmentFunction);
    
    MTL::AutoreleasedRenderPipelineReflection reflection = nullptr; // MTL::RenderPipelineReflection::alloc()->init();
    try
    {
        m_MeshPipeline = metalDevice->newRenderPipelineState(meshDesc.get(), MTL::PipelineOptionNone, &reflection, &error);
    }
    catch (std::exception& ex)
    {
        ::fprintf(stderr, "%s", ex.what());
    }
    
    if (error != nullptr)
    {
        ::fprintf(stderr, "Metal: Error creating pipeline state: %s\n%s\n", error->localizedDescription()->utf8String(), error->localizedFailureReason()->utf8String());
        error = nullptr;
    }
}


RenderAPI_Metal::RenderAPI_Metal()
{
}


void RenderAPI_Metal::ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces)
{
	if (type == kUnityGfxDeviceEventInitialize)
	{
		m_MetalGraphics = interfaces->Get<IUnityGraphicsMetal>();

		CreateResources();
	}
	else if (type == kUnityGfxDeviceEventShutdown)
	{
		//@TODO: release resources
	}
}


void RenderAPI_Metal::DrawSimpleTriangles(const float worldMatrix[16], int triangleCount, const void* verticesFloat3Byte4)
{
	// Update vertex and constant buffers
	//@TODO: we don't do any synchronization here :)

	const int vbSize = triangleCount * 3 * kVertexSize;
	const int cbSize = 16 * sizeof(float);

	::memcpy(m_VertexBuffer->contents(), verticesFloat3Byte4, vbSize);
	::memcpy(m_ConstantBuffer->contents(), worldMatrix, cbSize);

#if UNITY_OSX
	m_VertexBuffer->didModifyRange(NS::Range(0, vbSize));
	m_ConstantBuffer->didModifyRange(NS::Range(0, cbSize));
#endif

	MTL::RenderCommandEncoder* cmd = (MTL::RenderCommandEncoder*)m_MetalGraphics->CurrentCommandEncoder();

	// Setup rendering state
	cmd->setRenderPipelineState(m_Pipeline);
	cmd->setDepthStencilState(m_DepthStencil);
	cmd->setCullMode(MTL::CullModeNone);

	// Bind buffers
	cmd->setVertexBuffer(m_VertexBuffer, 0, 1);
	cmd->setVertexBuffer(m_ConstantBuffer, 0, 0);

	// Draw
	cmd->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), triangleCount*3);
}

void RenderAPI_Metal::DrawMesh(const float worldMatrix[16], void* positionBuffer, void* colorBuffer, int count)
{
    const int pbSize = 3 * 16;
    const int colorbSize = 3 * 16;
    const int cbSize = 16 * sizeof(float);
    
    ::memcpy(m_ConstantBuffer->contents(), worldMatrix, cbSize);
    ::memcpy(m_PositionBuffer->contents(), positionBuffer, pbSize);
    ::memcpy(m_ColorBuffer->contents(), colorBuffer, colorbSize);
    
#if UNITY_OSX
    m_PositionBuffer->didModifyRange(NS::Range(0, pbSize));
    m_ColorBuffer->didModifyRange(NS::Range(0, colorbSize));
    m_ConstantBuffer->didModifyRange(NS::Range(0, cbSize));
#endif
    
    MTL::RenderCommandEncoder* cmd = (MTL::RenderCommandEncoder*)m_MetalGraphics->CurrentCommandEncoder();
    
    cmd->setRenderPipelineState(m_MeshPipeline);
    cmd->setDepthStencilState(m_DepthStencil);
    cmd->setCullMode(MTL::CullModeNone);
    
    cmd->setMeshBuffer(m_ConstantBuffer, 0, 0);
    cmd->setMeshBuffer(m_PositionBuffer, 0, 1);
    cmd->setMeshBuffer(m_ColorBuffer, 0, 2);
    
    cmd->drawMeshThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}


void* RenderAPI_Metal::BeginModifyTexture(void* textureHandle, int textureWidth, int textureHeight, int* outRowPitch)
{
	const int rowPitch = textureWidth * 4;
	// Just allocate a system memory buffer here for simplicity
	unsigned char* data = new unsigned char[rowPitch * textureHeight];
	*outRowPitch = rowPitch;
	return data;
}


void RenderAPI_Metal::EndModifyTexture(void* textureHandle, int textureWidth, int textureHeight, int rowPitch, void* dataPtr)
{
	MTL::Texture* tex = (MTL::Texture*)textureHandle;
	// Update texture data, and free the memory buffer
	tex->replaceRegion(MTL::Region(0,0,0, textureWidth,textureHeight,1), 0, dataPtr, rowPitch);
	delete[](unsigned char*)dataPtr;
}


void* RenderAPI_Metal::BeginModifyVertexBuffer(void* bufferHandle, size_t* outBufferSize)
{
	MTL::Buffer* buf = (MTL::Buffer*)bufferHandle;
	*outBufferSize = buf->length();
	return buf->contents();
}


void RenderAPI_Metal::EndModifyVertexBuffer(void* bufferHandle)
{
#	if UNITY_OSX
	MTL::Buffer* buf = (MTL::Buffer*)bufferHandle;
	buf->didModifyRange(NS::Range(0, buf->length()));
#	endif // if UNITY_OSX
}


#endif // #if SUPPORT_METAL
