# RayTrophi Project Renaming Guide

## 🎯 Objective
Rename project from `raytrac_sdl2` to `RayTrophi` professionally without breaking:
- Git history
- Visual Studio 2022 project
- CMake build system
- File references

---

## 📋 Step-by-Step Renaming Plan

### Phase 1: Preparation (Clean Working Directory)

1. **Commit or stash current changes**
   ```bash
   cd "E:\visual studio proje c++\raytracing_Proje_Moduler"
   git add raytrac_sdl2/source/
   git commit -m "Work in progress - before renaming"
   ```

2. **Close Visual Studio** (IMPORTANT!)
   - VS locks files, causing issues

---

### Phase 2: Git-Safe Folder Rename

3. **Use git mv (preserves history)**
   ```bash
   git mv raytrac_sdl2 RayTrophi
   ```
   
   This is better than manual rename because:
   - ✅ Git tracks it as a rename, not delete+add
   - ✅ Preserves file history
   - ✅ Smaller commit diff

---

### Phase 3: Update Visual Studio Project Files

4. **Rename .vcxproj file**
   ```bash
   cd RayTrophi
   git mv raytrac_sdl2.vcxproj RayTrophi.vcxproj
   git mv raytrac_sdl2.vcxproj.filters RayTrophi.vcxproj.filters
   git mv raytrac_sdl2.vcxproj.user RayTrophi.vcxproj.user
   git mv raytrac_sdl2.vcxproj.maxkemal.nvuser RayTrophi.vcxproj.maxkemal.nvuser
   ```

5. **Update project GUID and names inside .vcxproj**
   
   Edit `RayTrophi.vcxproj`:
   - Find: `raytrac_sdl2`
   - Replace with: `RayTrophi`
   
   Key sections to update:
   ```xml
   <ProjectName>RayTrophi</ProjectName>
   <RootNamespace>RayTrophi</RootNamespace>
   <TargetName>RayTrophi</TargetName>
   ```

6. **Update .vcxproj.filters**
   
   Usually no changes needed unless raytrac_sdl2 is hardcoded.

---

### Phase 4: Update CMakeLists.txt

7. **Edit CMakeLists.txt**
   
   ```cmake
   # OLD:
   project(raytrac_sdl2 LANGUAGES CXX CUDA)
   
   # NEW:
   project(RayTrophi LANGUAGES CXX CUDA)
   ```
   
   Also update:
   ```cmake
   # OLD:
   add_executable(raytrac_sdl2 ${SOURCES})
   
   # NEW:
   add_executable(RayTrophi ${SOURCES})
   ```

---

### Phase 5: Update Source Code References

8. **Search for hardcoded paths**
   
   Files that might reference `raytrac_sdl2`:
   - `resource.h`
   - `*.rc` files
   - Any build scripts
   - Documentation

   Search and replace:
   ```
   Find: raytrac_sdl2
   Replace: RayTrophi
   ```

---

### Phase 6: Update README.md

9. **Update paths in README**
   
   ```markdown
   # OLD:
   cd RayTrophi/raytrac_sdl2
   
   # NEW:
   cd RayTrophi/RayTrophi
   ```

---

### Phase 7: Commit Changes

10. **Commit the rename**
    ```bash
    git add .
    git commit -m "Rename project from raytrac_sdl2 to RayTrophi
    
    - Renamed project folder (raytrac_sdl2 → RayTrophi)
    - Renamed VS project files (.vcxproj)
    - Updated CMakeLists.txt project name
    - Updated all internal references
    - Updated README paths
    
    This maintains git history while establishing professional naming."
    ```

11. **Push to GitHub**
    ```bash
    git push origin main
    ```

---

### Phase 8: Verification

12. **Test Visual Studio build**
    - Open `RayTrophi.vcxproj` in VS2022
    - Build → Should succeed
    - Output should be: `RayTrophi.exe` (not `raytrac_sdl2.exe`)

13. **Test CMake build** (if used)
    ```bash
    cd build
    cmake ..
    cmake --build . --config Release
    ```

14. **Verify executable name**
    - Should be `RayTrophi.exe` or `raytracing_render_code.exe`
    - Update `TargetName` in .vcxproj if needed

---

## ⚠️ Important Notes

### What Git Preserves
✅ Full file history  
✅ Blame annotations  
✅ Commit graph  

### What Changes
⚠️ Clone URLs (same)  
⚠️ Local paths (update your shortcuts)  
⚠️ Build output directory names  

### Potential Issues & Solutions

**Issue 1: VS2022 can't find project**
- Solution: Delete `build/` folder, rebuild

**Issue 2: CMake cache errors**
- Solution: `rm -rf build/` and re-run cmake

**Issue 3: Git shows as delete + add**
- Solution: Use `git mv` (not manual rename)

---

## 🎯 Expected Final Structure

```
RayTrophi/
├── .gitignore
├── README.md
├── README_TR.md
├── RayTrophi/                    ← RENAMED FROM raytrac_sdl2
│   ├── RayTrophi.vcxproj        ← RENAMED FROM raytrac_sdl2.vcxproj
│   ├── RayTrophi.vcxproj.filters
│   ├── CMakeLists.txt            ← UPDATED: project(RayTrophi)
│   ├── source/
│   │   ├── cpp_file/
│   │   ├── header/
│   │   └── ...
│   ├── build/                    ← Can be deleted, will regenerate
│   ├── x64/
│   │   └── Release/
│   │       └── RayTrophi.exe    ← NEW NAME
│   └── ...
├── render_samples/
└── vcpkg/
```

---

## 🚀 Quick Commands (Copy-Paste)

```bash
# 1. Commit current work
cd "E:\visual studio proje c++\raytracing_Proje_Moduler"
git add raytrac_sdl2/source/
git commit -m "WIP: Before project rename"

# 2. Rename folder
git mv raytrac_sdl2 RayTrophi

# 3. Rename project files
cd RayTrophi
git mv raytrac_sdl2.vcxproj RayTrophi.vcxproj
git mv raytrac_sdl2.vcxproj.filters RayTrophi.vcxproj.filters
git mv raytrac_sdl2.vcxproj.user RayTrophi.vcxproj.user 2>$null
git mv raytrac_sdl2.vcxproj.maxkemal.nvuser RayTrophi.vcxproj.maxkemal.nvuser 2>$null

# 4. Update references (manual: edit .vcxproj and CMakeLists.txt)

# 5. Commit rename
cd ..
git add .
git commit -m "Rename project: raytrac_sdl2 → RayTrophi"
git push origin main
```

---

## ✅ Checklist

Before rename:
- [ ] Close Visual Studio
- [ ] Commit/stash changes
- [ ] Backup (optional but recommended)

During rename:
- [ ] Use `git mv` for folders
- [ ] Use `git mv` for .vcxproj files
- [ ] Update .vcxproj content
- [ ] Update CMakeLists.txt
- [ ] Update README.md

After rename:
- [ ] Test VS2022 build
- [ ] Test CMake build (optional)
- [ ] Verify executable name
- [ ] Push to GitHub
- [ ] Update local shortcuts/paths

---

**Time estimate: 15-20 minutes**  
**Risk level: Low (git preserves history)**  
**Difficulty: Medium**

Ready to proceed? 🚀
