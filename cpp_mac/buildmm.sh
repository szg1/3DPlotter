clang++ sips.mm -o PlotMetal \
  -std=c++17 -fobjc-arc \
  -framework Cocoa -framework Metal -framework MetalKit -framework QuartzCore
./PlotMetal

