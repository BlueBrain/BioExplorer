The Multiview core plug-in
==========================

This module implements the multi-view (top/front/right/perspective) camera

Usage
-----

- Point LD_LIBRARY_PATH to the folder which contains
  'CorePluginMultiview.so'
- Run Brayns application either with command line '--module CorePluginMultiview --camera-type multiview' or do
  'ospLoadModule("CorePluginMultiview")' programmatically
```
OSPCamera camera = ospNewCamera("multiview");
```
