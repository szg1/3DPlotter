#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/CAMetalLayer.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

// ------------------------------ Math helpers ------------------------------
struct float3 { float x,y,z; };
struct float4 { float x,y,z,w; };
struct float4x4 { float m[16]; }; // column-major

static inline float4x4 identity() {
    float4x4 M = { {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    } }; return M;
}
static inline float4x4 translation(float tx,float ty,float tz){
    float4x4 M = identity();
    M.m[12]=tx; M.m[13]=ty; M.m[14]=tz;
    return M;
}
static inline float4x4 rotationZ(float deg){
    float r=deg*(float)M_PI/180.0f; float c=cosf(r), s=sinf(r);
    float4x4 M = { {
        c, s, 0, 0,
       -s, c, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    } }; return M;
}
static inline float4x4 perspective(float fovyDeg,float aspect,float nearZ,float farZ){
    float f = 1.0f/tanf((fovyDeg*(float)M_PI/180.0f)*0.5f);
    float A = (farZ)/(nearZ - farZ);
    float4x4 M = { {
        f/aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, A, -1,
        0, 0, A*nearZ, 0
    } }; return M;
}
static inline float dot3(const float3&a,const float3&b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static inline float3 cross3(const float3&a,const float3&b){
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
static inline float3 normalize3(const float3&v){
    float L = std::max(1e-6f, std::sqrt(dot3(v,v)));
    return { v.x/L, v.y/L, v.z/L };
}
static inline float4x4 mul(const float4x4&A,const float4x4&B){
    float4x4 R{};
    for(int c=0;c<4;++c) for(int r=0;r<4;++r){
        R.m[c*4+r] =
            A.m[0*4+r]*B.m[c*4+0] +
            A.m[1*4+r]*B.m[c*4+1] +
            A.m[2*4+r]*B.m[c*4+2] +
            A.m[3*4+r]*B.m[c*4+3];
    }
    return R;
}

// ------------------------------ Scene data ------------------------------
struct Pt { int x, y; float z; };

static inline bool feasible(int x,int y){
    // x >= 0, y >= 0, x <= 40, 2x + y <= 100, x + y <= 80
    return (x>=0) && (y>=0) && (x<=40) && (2*x + y <= 100) && (x+y <= 80);
}
static inline float surface(float x,float y){ return 3.0f*x + 2.0f*y; }

struct BBox { float minx,maxx,miny,maxy,minz,maxz; };

struct Uniforms {
    float4x4 mvp;
    float4   color;
    float    pointSize; // in world units for billboards
    float    pad0, pad1, pad2;
};

// ------------------------------ Metal shaders (runtime source) ------------------------------
static NSString* const kShaderSrc = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct VSIn {
    float3 pos [[attribute(0)]];
    float3 nrm [[attribute(1)]];
};

struct VSOut {
    float4 pos [[position]];
    float3 nrm;
    float3 worldPos;
};

struct Uniforms {
    float4x4 mvp;
    float4   color;
    float    pointSize;
    float    pad0, pad1, pad2;
};

// ---------- Surface & Lines ----------
vertex VSOut vs_main(uint vid [[vertex_id]],
                     const device VSIn* vbuf [[buffer(0)]],
                     constant Uniforms& U [[buffer(1)]])
{
    VSOut o;
    float3 p = vbuf[vid].pos;
    float3 n = vbuf[vid].nrm;
    o.worldPos = p;
    o.nrm = n;
    o.pos = U.mvp * float4(p,1.0);
    return o;
}

fragment float4 fs_lit(VSOut in [[stage_in]],
                       constant Uniforms& U [[buffer(1)]])
{
    // simple lambert lighting with fixed light dir in world
    float3 L = normalize(float3(0.5,0.6,1.0));
    float ndl = max(0.1, dot(normalize(in.nrm), L));
    return float4(U.color.rgb * ndl, U.color.a);
}

fragment float4 fs_solid(VSOut in [[stage_in]],
                         constant Uniforms& U [[buffer(1)]])
{
    return U.color;
}

// ---------- Instanced point billboards ----------
struct VSInPoint {
    float3 center; // per-instance center position
};
struct VSOutPoint {
    float4 pos [[position]];
    float2 uv;
};
vertex VSOutPoint vs_point(uint vid [[vertex_id]],
                           uint iid [[instance_id]],
                           const device VSInPoint* pbuf [[buffer(0)]],
                           constant Uniforms& U [[buffer(1)]])
{
    // unit quad vertices (-0.5..+0.5) in a triangle strip order
    float2 quad[4] = { float2(-0.5,-0.5), float2(0.5,-0.5),
                       float2(-0.5,0.5),  float2(0.5,0.5) };
    float2 q = quad[vid];
    float3 center = pbuf[iid].center;

    // build a small square in world space facing camera (billboard) using world XY (top-down camera)
    float size = U.pointSize;
    float3 p = center + float3(q.x*size, q.y*size, 0.0);

    VSOutPoint o;
    o.pos = U.mvp * float4(p,1.0);
    o.uv = q + 0.5; // 0..1
    return o;
}
fragment float4 fs_point(VSOutPoint in [[stage_in]],
                         constant Uniforms& U [[buffer(1)]])
{
    // round point mask
    float2 d = in.uv - 0.5;
    float r2 = dot(d,d);
    if (r2 > 0.25) discard_fragment;
    return U.color;
}
)METAL";

// ------------------------------ Renderer ------------------------------
@interface Renderer : NSObject<MTKViewDelegate>
@end

@implementation Renderer {
    id<MTLDevice>        _device;
    id<MTLCommandQueue>  _queue;
    id<MTLLibrary>       _lib;

    id<MTLRenderPipelineState> _psoSurface;
    id<MTLRenderPipelineState> _psoLines;
    id<MTLRenderPipelineState> _psoPoints;
    id<MTLDepthStencilState>   _dss;

    // Buffers
    id<MTLBuffer> _vbSurface;   // interleaved pos/nrm
    id<MTLBuffer> _vbLines;     // pos only, nrm dummy
    id<MTLBuffer> _vbPointInst; // per-instance centers
    id<MTLBuffer> _ubuf;        // uniforms

    // Geometry counts
    NSUInteger _surfaceVertCount = 0;
    NSUInteger _linesVertCount   = 0;
    NSUInteger _pointCount       = 0;

    // Scene parameters
    std::vector<Pt> _points;
    BBox   _bbox;
    float  _cx,_cy,_cz;
    float  _radius;
    float  _fovY = 45.0f;
    float  _margin = 1.15f;
    float  _elevDeg = 30.0f;      // camera elevation
    float  _camDist = 200.0f;     // computed
    float  _aspect  = 1.0f;
    float  _modelAngle = 0.0f;    // degrees
    CFAbsoluteTime _lastTime;
}

- (instancetype)initWithView:(MTKView*)view
{
    if (self = [super init]) {
        _device = view.device;
        _queue  = [_device newCommandQueue];

        NSError* err = nil;
        _lib = [_device newLibraryWithSource:kShaderSrc options:nil error:&err];
        if (!_lib) { NSLog(@"Library error: %@", err); exit(1); }

        // Pipelines
        MTLRenderPipelineDescriptor* srf = [MTLRenderPipelineDescriptor new];
        srf.vertexFunction   = [_lib newFunctionWithName:@"vs_main"];
        srf.fragmentFunction = [_lib newFunctionWithName:@"fs_lit"];
        srf.colorAttachments[0].pixelFormat = view.colorPixelFormat;
        srf.colorAttachments[0].blendingEnabled = YES;
        srf.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        srf.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        srf.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        srf.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        srf.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        srf.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        srf.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
        _psoSurface = [_device newRenderPipelineStateWithDescriptor:srf error:&err];
        NSCAssert(_psoSurface, @"Surface PSO error: %@", err);

        MTLRenderPipelineDescriptor* lin = [MTLRenderPipelineDescriptor new];
        lin.vertexFunction   = [_lib newFunctionWithName:@"vs_main"];
        lin.fragmentFunction = [_lib newFunctionWithName:@"fs_solid"];
        lin.colorAttachments[0].pixelFormat = view.colorPixelFormat;
        lin.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
        _psoLines = [_device newRenderPipelineStateWithDescriptor:lin error:&err];
        NSCAssert(_psoLines, @"Lines PSO error: %@", err);

        MTLRenderPipelineDescriptor* pnt = [MTLRenderPipelineDescriptor new];
        pnt.vertexFunction   = [_lib newFunctionWithName:@"vs_point"];
        pnt.fragmentFunction = [_lib newFunctionWithName:@"fs_point"];
        pnt.colorAttachments[0].pixelFormat = view.colorPixelFormat;
        pnt.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
        pnt.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        _psoPoints = [_device newRenderPipelineStateWithDescriptor:pnt error:&err];
        NSCAssert(_psoPoints, @"Points PSO error: %@", err);

        // Depth state
        MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
        ds.depthCompareFunction = MTLCompareFunctionLess;
        ds.depthWriteEnabled = YES;
        _dss = [_device newDepthStencilStateWithDescriptor:ds];

        // Build geometry from constraints
        [self buildScene];

        // Uniforms buffer
        _ubuf = [_device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];

        _lastTime = CFAbsoluteTimeGetCurrent();
    }
    return self;
}

// ---- Build data: points, surface mesh, axes (with simple "X/Y/Z" glyphs) ----
- (void)buildScene {
    // Integer-feasible points
    _points.clear();
    for (int x=0; x<=40; ++x)
        for (int y=0; y<=40; ++y)
            if (feasible(x,y))
                _points.push_back({x,y,(float)surface(x,y)});

    // Bounds
    _bbox = { 1e9f,-1e9f, 1e9f,-1e9f, 1e9f,-1e9f };
    for (const auto& p : _points) {
        _bbox.minx = std::min(_bbox.minx,(float)p.x);
        _bbox.maxx = std::max(_bbox.maxx,(float)p.x);
        _bbox.miny = std::min(_bbox.miny,(float)p.y);
        _bbox.maxy = std::max(_bbox.maxy,(float)p.y);
        _bbox.minz = std::min(_bbox.minz,(float)p.z);
        _bbox.maxz = std::max(_bbox.maxz,(float)p.z);
    }
    _cx = 0.5f*(_bbox.minx+_bbox.maxx);
    _cy = 0.5f*(_bbox.miny+_bbox.maxy);
    _cz = 0.5f*(_bbox.minz+_bbox.maxz);
    float dx=_bbox.maxx-_bbox.minx, dy=_bbox.maxy-_bbox.miny, dz=_bbox.maxz-_bbox.minz;
    _radius = 0.5f*std::sqrt(dx*dx+dy*dy+dz*dz);
    if (_radius<1e-3f) _radius=1.0f;

    // --- Surface mesh (triangle list), normal is constant for z=3x+2y ---
    const float3 nrm = normalize3({-3.0f,-2.0f,1.0f});
    struct V { float3 pos; float3 nrm; };
    std::vector<V> srf;

    for (int x=0; x<40; ++x) for (int y=0; y<40; ++y) {
        int x0=x, x1=x+1, y0=y, y1=y+1;
        if (x1>40 || y1>40) continue;
        if (feasible(x0,y0) && feasible(x1,y0) && feasible(x1,y1) && feasible(x0,y1)) {
            float z00=surface(x0,y0), z10=surface(x1,y0);
            float z11=surface(x1,y1), z01=surface(x0,y1);
            // two triangles
            srf.push_back({{(float)x0,(float)y0,z00}, nrm});
            srf.push_back({{(float)x1,(float)y0,z10}, nrm});
            srf.push_back({{(float)x1,(float)y1,z11}, nrm});

            srf.push_back({{(float)x0,(float)y0,z00}, nrm});
            srf.push_back({{(float)x1,(float)y1,z11}, nrm});
            srf.push_back({{(float)x0,(float)y1,z01}, nrm});
        }
    }
    _surfaceVertCount = srf.size();
    _vbSurface = [_device newBufferWithBytes:srf.data()
                                      length:srf.size()*sizeof(V)
                                     options:MTLResourceStorageModeManaged];

    // --- Lines: axes through bbox center + simple "X/Y/Z" glyphs made of short segments ---
    std::vector<V> lines;
    auto pushSeg = [&](float3 a, float3 b, float3 colorNrm){
        lines.push_back({a, colorNrm});
        lines.push_back({b, colorNrm});
    };

    float axisLen = std::max({dx,dy,dz}); axisLen = std::max(10.0f, axisLen*0.8f + 10.0f);
    float3 cxp = {_cx,_cy,_cz};

    // Axes (store axis color in "nrm" to pick color in shader without extra buffers)
    // X: red (1,0,0)
    pushSeg({cxp.x-axisLen, cxp.y, cxp.z}, {cxp.x+axisLen, cxp.y, cxp.z}, {1,0,0});
    // Y: green-ish (0,0.6,0)
    pushSeg({cxp.x, cxp.y-axisLen, cxp.z}, {cxp.x, cxp.y+axisLen, cxp.z}, {0,0.6f,0});
    // Z: blue (0,0,1)
    pushSeg({cxp.x, cxp.y, cxp.z-axisLen}, {cxp.x, cxp.y, cxp.z+axisLen}, {0,0,1});

    // Glyphs near tips using tiny segments ("X","Y","Z")
    auto addGlyphX = [&](float3 tip){
        float s=axisLen*0.06f;
        pushSeg({tip.x-s,tip.y+s,tip.z},{tip.x+s,tip.y-s,tip.z},{1,0,0});
        pushSeg({tip.x-s,tip.y-s,tip.z},{tip.x+s,tip.y+s,tip.z},{1,0,0});
    };
    auto addGlyphY = [&](float3 tip){
        float s=axisLen*0.06f;
        pushSeg({tip.x,tip.y,tip.z},{tip.x,tip.y-s,tip.z},{0,0.6f,0});
        pushSeg({tip.x,tip.y,tip.z},{tip.x-s,tip.y+s,tip.z},{0,0.6f,0});
        pushSeg({tip.x,tip.y,tip.z},{tip.x+s,tip.y+s,tip.z},{0,0.6f,0});
    };
    auto addGlyphZ = [&](float3 tip){
        float s=axisLen*0.06f;
        pushSeg({tip.x-s,tip.y+s,tip.z},{tip.x+s,tip.y+s,tip.z},{0,0,1});
        pushSeg({tip.x+s,tip.y+s,tip.z},{tip.x-s,tip.y-s,tip.z},{0,0,1});
        pushSeg({tip.x-s,tip.y-s,tip.z},{tip.x+s,tip.y-s,tip.z},{0,0,1});
    };
    addGlyphX({cxp.x+axisLen*1.05f, cxp.y, cxp.z});
    addGlyphY({cxp.x, cxp.y+axisLen*1.05f, cxp.z});
    addGlyphZ({cxp.x, cxp.y, cxp.z+axisLen*1.05f});

    _linesVertCount = lines.size();
    _vbLines = [_device newBufferWithBytes:lines.data()
                                    length:lines.size()*sizeof(V)
                                   options:MTLResourceStorageModeManaged];

    // --- Point instances: centers only ---
    struct P { float3 c; };
    std::vector<P> inst; inst.reserve(_points.size());
    for (auto& p : _points) inst.push_back({{(float)p.x,(float)p.y,p.z}});
    _pointCount = inst.size();
    _vbPointInst = [_device newBufferWithBytes:inst.data()
                                        length:inst.size()*sizeof(P)
                                       options:MTLResourceStorageModeManaged];
}

// Compute camera distance to fit bounding sphere (auto-framing)
- (void)computeCameraForAspect:(float)aspect {
    _aspect = aspect;
    float fovy = _fovY * (float)M_PI/180.0f;
    float fovx = 2.f * atanf(tanf(fovy*0.5f) * _aspect);
    auto need = [&](float fov){ return _radius / std::max(0.5f, sinf(fov*0.5f)); };
    float d = std::max(need(fovy), need(fovx));
    _camDist = d * _margin;
    if (_camDist < 5.f) _camDist = 5.f;
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    float aspect = (float)size.width / std::max(1.f,(float)size.height);
    [self computeCameraForAspect:aspect];
}

- (void)drawInMTKView:(MTKView *)view {
    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    float dt = (float)(now - _lastTime);
    _lastTime = now;
    _modelAngle += 30.f * dt; // deg/sec
    if (_modelAngle > 360.f) _modelAngle -= 360.f;

    id<CAMetalDrawable> drawable = view.currentDrawable;
    MTLRenderPassDescriptor* rp = view.currentRenderPassDescriptor;
    if (!drawable || !rp) return;

    // --- Build MVP ---
    float elevRad = 30.0f*(float)M_PI/180.0f;
    // Eye at (cx, cy - camDist*cos(elev), cz + camDist*sin(elev))
    float3 eye = { _cx, _cy - _camDist*cosf(elevRad), _cz + _camDist*sinf(elevRad) };

    // Build a simple lookAt matrix (world->view)
    float3 F = { _cx - eye.x, _cy - eye.y, _cz - eye.z };
    F = normalize3(F);
    float3 up = {0,0,1};
    float3 S = normalize3(cross3(F, up));
    float3 U = cross3(S, F);

    float4x4 viewM = identity();
    viewM.m[0]= S.x; viewM.m[4]= S.y; viewM.m[8 ]= S.z;
    viewM.m[1]= U.x; viewM.m[5]= U.y; viewM.m[9 ]= U.z;
    viewM.m[2]=-F.x; viewM.m[6]=-F.y; viewM.m[10]=-F.z;
    // translation
    float4x4 T = translation(-eye.x, -eye.y, -eye.z);
    viewM = mul(T, viewM);

    // Model rotates about center
    float4x4 model = mul( translation(_cx,_cy,_cz), mul(rotationZ(_modelAngle), translation(-_cx,-_cy,-_cz)) );

    float aspect = (float)view.drawableSize.width / std::max(1.f,(float)view.drawableSize.height);
    [self computeCameraForAspect:aspect];
    float4x4 proj = perspective(45.0f, aspect, 0.5f, 5000.0f);

    float4x4 mvp = mul(model, identity());
    mvp = mul(viewM, mvp);
    mvp = mul(proj, mvp);

    // Uniforms
    Uniforms* U = (Uniforms*)_ubuf.contents;
    U->mvp = mvp;
    U->color = {0.1f,0.4f,0.9f,0.6f};
    U->pointSize = 0.8f; // world size for point billboard

    id<MTLCommandBuffer> cb = [_queue commandBuffer];

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setDepthStencilState:_dss];

    // Surface
    [enc setRenderPipelineState:_psoSurface];
    [enc setVertexBuffer:_vbSurface offset:0 atIndex:0];
    [enc setVertexBuffer:_ubuf      offset:0 atIndex:1];
    [enc setFragmentBuffer:_ubuf    offset:0 atIndex:1];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:(NSUInteger)_surfaceVertCount];

    // Axes & glyph lines: reuse vs_main but color from U; encode axis color via separate draw calls
    // We'll draw them grey so the colored surface/points stand out; or we can grab per-vertex "nrm" as color — simpler: just set a uniform color here.
    U->color = {0,0,0,1};
    [enc setRenderPipelineState:_psoLines];
    [enc setVertexBuffer:_vbLines offset:0 atIndex:0];
    [enc setVertexBuffer:_ubuf    offset:0 atIndex:1];
    [enc setFragmentBuffer:_ubuf  offset:0 atIndex:1];
    [enc drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:(NSUInteger)_linesVertCount];

    // Points as instanced billboards
    U->color = {1,0,0,1};
    [enc setRenderPipelineState:_psoPoints];
    [enc setVertexBuffer:_vbPointInst offset:0 atIndex:0];
    [enc setVertexBuffer:_ubuf        offset:0 atIndex:1];
    // Draw a quad (4 verts) instanced _pointCount times -> triangle strip 2x2 => 4 verts, but Metal requires triangle list, so draw 4 verts in strip form is OK because we set topology class to triangle and use 4 verts -> but we actually rely on fragment discard to get round shape. We'll draw as triangle strip by issuing 4 verts per instance.
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:(NSUInteger)_pointCount];

    [enc endEncoding];
    [cb presentDrawable:drawable];
    [cb commit];
}

@end

// ------------------------------ App / View setup ------------------------------
@interface AppDelegate : NSObject <NSApplicationDelegate>
@property (strong) NSWindow* window;
@property (strong) MTKView*  view;
@property (strong) Renderer* renderer;
@end

@implementation AppDelegate
- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    NSRect frame = NSMakeRect(100, 100, 1000, 750);
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:NSWindowStyleMaskTitled |
                                                        NSWindowStyleMaskClosable |
                                                        NSWindowStyleMaskMiniaturizable |
                                                        NSWindowStyleMaskResizable
                                                backing:NSBackingStoreBuffered defer:NO];
    self.window.title = @"Integer-feasible surface (Metal, auto-framed, rotating)";

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    self.view = [[MTKView alloc] initWithFrame:frame device:dev];
    self.view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    self.view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    self.view.clearColor = MTLClearColorMake(1,1,1,1);
    self.view.preferredFramesPerSecond = 60;

    self.renderer = [[Renderer alloc] initWithView:self.view];
    self.view.delegate = self.renderer;

    [self.window setContentView:self.view];
    [self.window makeKeyAndOrderFront:nil];
}
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender { return YES; }
@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        AppDelegate* delegate = [AppDelegate new];
        [app setDelegate:delegate];
        [app run];
    }
    return 0;
}

