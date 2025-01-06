#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <algorithm>
#import <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
namespace stb_image {
	#import "./stb_image.h"
	#import "./stb_image_write.h"
}

id<MTLLibrary> metallib(id<MTLDevice> device, NSString *path) {
	__block id<MTLLibrary> library = nil;
	dispatch_fd_t fd = open([path UTF8String],O_RDONLY);
	NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
	long size = [[attributes objectForKey:NSFileSize] integerValue];
	if(size>0) {
		dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
		dispatch_read(fd,size,dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0),^(dispatch_data_t d, int e) {
			if(!e) {
				NSError *err = nil;
				library = [device newLibraryWithData:d error:&err];
			}
			close(fd);
			dispatch_semaphore_signal(semaphore);
		});
		dispatch_semaphore_wait(semaphore,DISPATCH_TIME_FOREVER);
	}
	return library;
}

int main(int argc, char *argv[]) {
	@autoreleasepool {
		
		int info[3];
		unsigned int *src = (unsigned int *)stb_image::stbi_load("./src.jpg",info,info+1,info+2,4);
		if(src) {
			
			int W = info[0];
			int H = info[1];
			unsigned int *dst = new unsigned int[W*H]; 
						
			id<MTLDevice> device = MTLCreateSystemDefaultDevice();
			
			MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO];
			descriptor.usage = MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
			id<MTLTexture> texture[2] = {
				[device newTextureWithDescriptor:descriptor],
				[device newTextureWithDescriptor:descriptor]
			};
			
			[texture[0] replaceRegion:MTLRegionMake2D(0,0,W,H) mipmapLevel:0 withBytes:src bytesPerRow:W<<2];
			
			id<MTLFunction> fxaa = [metallib(device,@"./FXAA3_11.metallib") newFunctionWithName:@"processimage"];
			id<MTLBuffer> resolution = [device newBufferWithLength:sizeof(float)*2 options:MTLResourceCPUCacheModeDefaultCache];
			float *res = (float *)[resolution contents];
			res[0] = W;
			res[1] = H;
			
			int tx = 1;
			int ty = 1;
			for(int k=1; k<5; k++) {
				if(W%(1<<k)==0) tx = 1<<k;
				if(H%(1<<k)==0) ty = 1<<k;
			}
			MTLSize threadGroupSize = MTLSizeMake(tx,ty,1);
			MTLSize threadGroups = MTLSizeMake(W/tx,W/ty,1);
			
			id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:fxaa error:nil];
			id<MTLCommandQueue> queue = [device newCommandQueue];
			id<MTLCommandBuffer> commandBuffer = queue.commandBuffer;
			id<MTLComputeCommandEncoder> encoder = commandBuffer.computeCommandEncoder;
			[encoder setComputePipelineState:pipelineState];
			[encoder setTexture:texture[0] atIndex:0];
			[encoder setTexture:texture[1] atIndex:1];
			[encoder setBuffer:resolution offset:0 atIndex:0];
			[encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize];
			[encoder endEncoding];
			[commandBuffer commit];
			[commandBuffer waitUntilCompleted];
			
			[texture[1] getBytes:dst bytesPerRow:W<<2 fromRegion:MTLRegionMake2D(0,0,W,H) mipmapLevel:0];

			stb_image::stbi_write_png("dst.png",W,H,4,(void const*)dst,W<<2);

			delete[] dst;
			delete[] src;
		}
	}
}