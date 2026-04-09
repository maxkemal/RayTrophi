Place matcap PNG/TGA/JPG files in this folder to make them available as default Matcap assets.

Recommended starter mats:

I cannot bundle third-party images here automatically. To add nice matcaps, download common matcap packs (search "matcap PNG pack") and copy desired files into this folder. After copying, use "Load Matcap..." in the Viewport UI to pick the file from disk (or copy into project and reopen the app).

Notes on viewport integration:
- The engine includes a Vulkan raster "Solid" viewport mode that supports Matcap shading for fast sculpt/paint feedback. Matcap files placed in this folder are automatically available in the Viewport Matcap picker.
- To use: add your matcap images (PNG/TGA/JPG) to this folder, then reload assets or restart the application and select Viewport → Shading → Matcap.