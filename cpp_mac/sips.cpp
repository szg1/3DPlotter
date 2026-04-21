#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

// ---------------- Config ----------------
static int   WIN_W = 1000;
static int   WIN_H = 750;
static float ROT_SPEED = 30.0f;     // degrees per second (model rotation)
static float FOV_Y_DEG = 45.0f;     // vertical field of view
static float ELEV_DEG  = 30.0f;     // camera elevation above horizon
static float MARGIN    = 1.15f;     // framing margin ( >1 expands view)
static int   SURF_STEP = 1;         // surface grid step (integer sampling)
static float SURFACE_ALPHA = 0.6f;
static float POINT_SIZE = 4.0f;
// ---------------------------------------

// integer-feasible points
struct Pt { int x, y; float z; };
std::vector<Pt> points;

// bounding box / sphere
struct BBox { float minx, maxx, miny, maxy, minz, maxz; };
BBox bbox;
float cx=0, cy=0, cz=0;    // center of bbox
float radiusBS=1.0f;       // bounding sphere radius
float camDist=200.0f;      // computed camera distance
float aspect= (float)WIN_W/(float)WIN_H;

// rotation state (model rotates around its own Z axis)
float modelAngleDeg = 0.0f;
int   lastTimeMs = 0;

// constraints and surface definition
inline bool feasible(int x, int y) {
    // Your constraints:
    // x >= 0, y >= 0, x <= 40, 2x + y <= 100, x + y <= 80
    return (x >= 0) && (y >= 0) && (x <= 40) && (2*x + y <= 100) && (x + y <= 80);
}
inline float surface(float x, float y) {
    return 3.0f * x + 2.0f * y; // z = 3x + 2y
}

void drawBitmapText3D(const std::string& s, void* font, float x, float y, float z) {
    glRasterPos3f(x, y, z);
    for (char c : s) glutBitmapCharacter(font, c);
}

void drawAxisArrow(float dirX, float dirY, float dirZ, float length,
                   float r, float g, float b, const char* label)
{
    glColor3f(r, g, b);
    glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(dirX*length, dirY*length, dirZ*length);
    glEnd();

    // simple "arrowhead" with three little lines
    const float ah = std::max(0.04f*length, 2.5f);
    const float at = ah*0.5f;
    float tipX = dirX*length, tipY = dirY*length, tipZ = dirZ*length;

    // build any perpendiculars
    float nx = (std::fabs(dirX) < 0.9f) ? 1.0f : 0.0f;
    float ny = (nx == 1.0f) ? 0.0f : 1.0f;
    float nz = 0.0f;
    float dl = std::sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);
    float nl = std::sqrt(nx*nx + ny*ny + nz*nz);
    nx/=nl; ny/=nl; // nz already 0
    // cross for another perp
    float cxp = dirY*nz - dirZ*ny;
    float cyp = dirZ*nx - dirX*nz;
    float czp = dirX*ny - dirY*nx;
    float cl = std::sqrt(cxp*cxp + cyp*cyp + czp*czp);
    if (cl > 1e-6f) { cxp/=cl; cyp/=cl; czp/=cl; }

    glBegin(GL_LINES);
        glVertex3f(tipX, tipY, tipZ);
        glVertex3f(tipX - dirX*ah + nx*at,  tipY - dirY*ah + ny*at,  tipZ - dirZ*ah + nz*at);
        glVertex3f(tipX, tipY, tipZ);
        glVertex3f(tipX - dirX*ah - nx*at,  tipY - dirY*ah - ny*at,  tipZ - dirZ*ah - nz*at);
        glVertex3f(tipX, tipY, tipZ);
        glVertex3f(tipX - dirX*ah + cxp*at, tipY - dirY*ah + cyp*at, tipZ - dirZ*ah + czp*at);
    glEnd();

    // label
    drawBitmapText3D(label, GLUT_BITMAP_HELVETICA_18,
                     tipX + 0.02f*length, tipY + 0.02f*length, tipZ + 0.02f*length);
}

void buildPoints() {
    points.clear();
    for (int x=0; x<=40; ++x) {
        for (int y=0; y<=40; ++y) {
            if (feasible(x,y)) {
                points.push_back({x, y, surface((float)x,(float)y)});
            }
        }
    }
}

void computeBounds() {
    if (points.empty()) {
        bbox = {0,1,0,1,0,1};
        cx = cy = cz = 0.5f;
        radiusBS = 1.0f;
        return;
    }
    bbox.minx = bbox.miny = bbox.minz =  1e9f;
    bbox.maxx = bbox.maxy = bbox.maxz = -1e9f;

    for (const auto& p : points) {
        bbox.minx = std::min(bbox.minx, (float)p.x);
        bbox.maxx = std::max(bbox.maxx, (float)p.x);
        bbox.miny = std::min(bbox.miny, (float)p.y);
        bbox.maxy = std::max(bbox.maxy, (float)p.y);
        bbox.minz = std::min(bbox.minz, (float)p.z);
        bbox.maxz = std::max(bbox.maxz, (float)p.z);
    }

    // Also consider the surface mesh corners on the feasible rectangle edges
    // (points already cover integer grid—good enough for our linear surface)

    cx = 0.5f*(bbox.minx + bbox.maxx);
    cy = 0.5f*(bbox.miny + bbox.maxy);
    cz = 0.5f*(bbox.minz + bbox.maxz);

    float dx = bbox.maxx - bbox.minx;
    float dy = bbox.maxy - bbox.miny;
    float dz = bbox.maxz - bbox.minz;
    radiusBS = 0.5f * std::sqrt(dx*dx + dy*dy + dz*dz);
    if (radiusBS < 1e-3f) radiusBS = 1.0f;
}

// compute camera distance to fit bounding sphere given FOV and aspect
void computeCameraFraming() {
    aspect = (float)WIN_W / (float)WIN_H;
    float fovyRad = FOV_Y_DEG * (float)M_PI / 180.0f;
    float fovxRad = 2.0f * std::atan(std::tan(fovyRad/2.0f) * aspect);

    // For a sphere of radius R, need D >= R / sin(FOV/2)
    float needY = radiusBS / std::sin(fovyRad/2.0f);
    float needX = radiusBS / std::sin(fovxRad/2.0f);
    float need = std::max(needX, needY);

    camDist = need * MARGIN;
    if (camDist < 5.0f) camDist = 5.0f; // safety floor
}

void initGL() {
    glEnable(GL_DEPTH_TEST);
    glPointSize(POINT_SIZE);
    glClearColor(1.0f,1.0f,1.0f,1.0f); // white bg

    // simple lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat lightPos[4] = { 200.0f, 200.0f, 300.0f, 1.0f };
    GLfloat lightCol[4] = { 0.9f, 0.9f, 0.9f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  lightCol);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightCol);

    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // set initial projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(FOV_Y_DEG, aspect, 0.5, 5000.0);
    glMatrixMode(GL_MODELVIEW);

    lastTimeMs = glutGet(GLUT_ELAPSED_TIME);
}

void drawAxes(float length) {
    // draw axes centered at bbox center
    glPushMatrix();
        glTranslatef(cx, cy, cz);
        drawAxisArrow(1,0,0, length, 1,0,0, "X");
        drawAxisArrow(0,1,0, length, 0,0.6f,0, "Y");
        drawAxisArrow(0,0,1, length, 0,0,1, "Z");
    glPopMatrix();
}

void drawIntegerPoints() {
    glDisable(GL_LIGHTING);
    glColor3f(1,0,0);
    glBegin(GL_POINTS);
    for (const auto& p : points) {
        glVertex3f((float)p.x, (float)p.y, p.z);
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

void drawSurfaceMesh() {
    glColor4f(0.1f, 0.4f, 0.9f, SURFACE_ALPHA);

    for (int x=0; x<40; x += SURF_STEP) {
        for (int y=0; y<40; y += SURF_STEP) {
            int x0 = x, x1 = x + SURF_STEP;
            int y0 = y, y1 = y + SURF_STEP;
            if (x1>40 || y1>40) continue;

            if (feasible(x0,y0) && feasible(x1,y0) && feasible(x1,y1) && feasible(x0,y1)) {
                float z00 = surface((float)x0,(float)y0);
                float z10 = surface((float)x1,(float)y0);
                float z11 = surface((float)x1,(float)y1);
                float z01 = surface((float)x0,(float)y1);

                // normal for lighting
                float ux = (float)(x1 - x0), uy = 0.0f, uz = (z10 - z00);
                float vx = 0.0f, vy = (float)(y1 - y0), vz = (z01 - z00);
                float nx = uy*vz - uz*vy;
                float ny = uz*vx - ux*vz;
                float nz = ux*vy - uy*vx;
                float nl = std::sqrt(nx*nx + ny*ny + nz*nz);
                if (nl > 1e-6f) { nx/=nl; ny/=nl; nz/=nl; }

                glBegin(GL_QUADS);
                    glNormal3f(nx, ny, nz);
                    glVertex3f((float)x0, (float)y0, z00);
                    glVertex3f((float)x1, (float)y0, z10);
                    glVertex3f((float)x1, (float)y1, z11);
                    glVertex3f((float)x0, (float)y1, z01);
                glEnd();
            }
        }
    }
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Camera position computed to frame bounding sphere, at given elevation.
    float elevRad = ELEV_DEG * (float)M_PI / 180.0f;
    float eyeX = cx;
    float eyeY = cy - camDist * std::cos(elevRad);
    float eyeZ = cz + camDist * std::sin(elevRad);

    gluLookAt(eyeX, eyeY, eyeZ,
              cx,   cy,   cz,
              0.0,  0.0,  1.0);

    // ---- rotate MODEL about its own Z axis around center (cx,cy,cz) ----
    glPushMatrix();
        glTranslatef(cx, cy, cz);
        glRotatef(modelAngleDeg, 0.0f, 0.0f, 1.0f);
        glTranslatef(-cx, -cy, -cz);

        float axisLen = std::max({bbox.maxx - bbox.minx, bbox.maxy - bbox.miny, bbox.maxz - bbox.minz});
        axisLen = std::max(axisLen, 1.0f) * 0.7f + 10.0f; // scale sensibly

        drawAxes(axisLen);
        drawSurfaceMesh();
        drawIntegerPoints();
    glPopMatrix();

    // static title (unrotated, in screen space)
    glDisable(GL_LIGHTING);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, WIN_W, 0, WIN_H);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix(); glLoadIdentity();
        glColor3f(0,0,0);
        glRasterPos2i(10, WIN_H - 24);
        const char* title = "z = 3x + 2y   constraints: x>=0, y>=0, x<=40, 2x+y<=100, x+y<=80";
        for (const char* p = title; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_LIGHTING);

    glutSwapBuffers();
}

void idle() {
    int now = glutGet(GLUT_ELAPSED_TIME);
    float dt = (now - lastTimeMs) / 1000.0f;
    lastTimeMs = now;

    modelAngleDeg += ROT_SPEED * dt;
    if (modelAngleDeg >= 360.0f) modelAngleDeg -= 360.0f;

    glutPostRedisplay();
}

void reshape(int w, int h) {
    WIN_W = std::max(w, 1);
    WIN_H = std::max(h, 1);
    glViewport(0, 0, WIN_W, WIN_H);

    aspect = (float)WIN_W / (float)WIN_H;

    // Recompute camera distance for new aspect
    computeCameraFraming();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(FOV_Y_DEG, aspect, 0.5, 5000.0);
    glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv) {
    // 1) Build data from constraints
    buildPoints();

    // 2) Compute bounding box & sphere
    computeBounds();

    // 3) Compute camera framing for initial window
    computeCameraFraming();

    // 4) GLUT / OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIN_W, WIN_H);
    glutCreateWindow("Integer-feasible surface (auto-framed, rotating model)");

    initGL();

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);

    glutMainLoop();
    return 0;
}

