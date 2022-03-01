#pragma once

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include "Exception.h"

#include <fstream>
#include <vector>

#include <direct.h>

struct Params
{
	uchar4* image;
	unsigned int image_width;
};

struct RayGenData
{
	float r, g, b;
};

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<int>        MissSbtRecord;

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}


OptixDeviceContext context = nullptr;
OptixModule module = nullptr;
OptixPipelineCompileOptions pipeline_compile_options = {};
OptixProgramGroup raygen_prog_group = nullptr;
OptixProgramGroup miss_prog_group = nullptr;
OptixPipeline pipeline = nullptr;
OptixShaderBindingTable sbt = {};

uchar4* device_pixels = nullptr;
std::vector<uchar4> host_pixels;
void Init()
{
	char log[2048]; // For error reporting from OptiX creation functions

	//
	// Initialize CUDA and create OptiX context
	//
	{
		// Initialize CUDA
		CUDA_CHECK(cudaFree(0));

		CUcontext cuCtx = 0;  // zero means take the current context
		OPTIX_CHECK(optixInit());
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
	}

	//
	// Create module
	//
	{
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

		pipeline_compile_options.usesMotionBlur = false;
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		pipeline_compile_options.numPayloadValues = 2;
		pipeline_compile_options.numAttributeValues = 2;
		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

		std::string currentPath(_getcwd(NULL, 0));
		std::string filename = currentPath + "/../PluginSource/build/x64/Debug/draw_solid_color.ptx";

		size_t inputSize = 0;
		std::fstream file(filename);
		std::string source(std::istreambuf_iterator<char>(file), {});
		const char* input = source.c_str();
		inputSize = source.size();

		size_t sizeof_log = sizeof(log);

		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			context,
			&module_compile_options,
			&pipeline_compile_options,
			input,
			inputSize,
			log,
			&sizeof_log,
			&module
		));
	}

	//
	// Create program groups, including NULL miss and hitgroups
	//
	{
		OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

		OptixProgramGroupDesc raygen_prog_group_desc = {}; //
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context,
			&raygen_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&raygen_prog_group
		));

		// Leave miss group's module and entryfunc name null
		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context,
			&miss_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&miss_prog_group
		));
	}

	//
	// Link pipeline
	//
	{
		const uint32_t    max_trace_depth = 0;
		OptixProgramGroup program_groups[] = { raygen_prog_group };

		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = max_trace_depth;
		pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixPipelineCreate(
			context,
			&pipeline_compile_options,
			&pipeline_link_options,
			program_groups,
			sizeof(program_groups) / sizeof(program_groups[0]),
			log,
			&sizeof_log,
			&pipeline
		));

		OptixStackSizes stack_sizes = {};
		for (auto& prog_group : program_groups)
		{
			OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
		}

		uint32_t direct_callable_stack_size_from_traversal;
		uint32_t direct_callable_stack_size_from_state;
		uint32_t continuation_stack_size;
		OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
			0,  // maxCCDepth
			0,  // maxDCDEpth
			&direct_callable_stack_size_from_traversal,
			&direct_callable_stack_size_from_state, &continuation_stack_size));
		OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
			direct_callable_stack_size_from_state, continuation_stack_size,
			2  // maxTraversableDepth
		));
	}

	//
	// Set up shader binding table
	//
	{
		CUdeviceptr  raygen_record;
		const size_t raygen_record_size = sizeof(RayGenSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
		RayGenSbtRecord rg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
		rg_sbt.data = { 0.462f, 0.725f, 0.f };
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(raygen_record),
			&rg_sbt,
			raygen_record_size,
			cudaMemcpyHostToDevice
		));

		CUdeviceptr miss_record;
		size_t      miss_record_size = sizeof(MissSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
		RayGenSbtRecord ms_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(miss_record),
			&ms_sbt,
			miss_record_size,
			cudaMemcpyHostToDevice
		));

		sbt.raygenRecord = raygen_record;
		sbt.missRecordBase = miss_record;
		sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
		sbt.missRecordCount = 1;
	}





}

int         width = 512;
int         height = 384;
uchar4* Launch(int _width, int _height)
{
	width = _width;
	height = _height;
	//
	// Create cuda device resource.
	//
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(device_pixels)));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&device_pixels),
		width * height * sizeof(uchar4)
	));

	CUstream stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	Params params;
	params.image = device_pixels;
	params.image_width = width;

	CUdeviceptr d_param;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_param),
		&params, sizeof(params),
		cudaMemcpyHostToDevice
	));

	OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
	CUDA_SYNC_CHECK();

	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaStreamSynchronize(0u));


	host_pixels.resize(width * height);
	CUDA_CHECK(cudaMemcpy(
		static_cast<void*>(host_pixels.data()),
		device_pixels,
		width * height * sizeof(uchar4),
		cudaMemcpyDeviceToHost
	));

	return host_pixels.data();
}

void Cleanup()
{
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));

	OPTIX_CHECK(optixPipelineDestroy(pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
	OPTIX_CHECK(optixModuleDestroy(module));

	OPTIX_CHECK(optixDeviceContextDestroy(context));
}