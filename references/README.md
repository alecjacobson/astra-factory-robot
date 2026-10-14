# Visual references

Reference imagery was searched and inspected on 14 October 2026. The model is original procedural geometry, inspired by the forms below; no manufacturer CAD or third-party photographs are embedded in the Blender file or redistributed here.

| Reference | Observed features used |
|---|---|
| [Mercedes W203 removed headliner photograph](https://mbworld.org/forums/c-class-w203/787955-w203-headliner-removal.html) · [image](https://cimg9.ibsrv.net/gimg/www.mbworld.org-vbulletin/2000x1504/img_7041_155f268984f38709425ceab950706b0ba5e1330d.jpg) | Broad molded roof panel; large sunroof aperture; overhead console, lamp, and smaller mounting cutouts; curved perimeter. |
| [KUKA KR Quantec 210 R2700 — Red Dot](https://www.red-dot.org/de/project/kuka-kr-quantec-210-r2700-45875) · [image](https://www.red-dot.org/fileadmin/user_upload/projects/2020/PD/34-05826-2020-3.jpg) | Orange cast links; dark pedestal; circular joint housings; tapered forearm; visible service conduit; wrist flange. |
| [MESH Automation: robotic load/unload of riveting machines](https://meshautomationinc.com/case_studies/robotic-load-unload-riveting-machines/) | Manufacturer/integrator example of a KUKA arm with lightweight aluminum vacuum tooling, transferring parts between nests. This is a tooling precedent, not a headliner installation. |

The headliner is a generic approximately 1.421 × 1.870 m passenger-car panel, not a dimensionally accurate reproduction of a particular vehicle. The robot is a generic articulated handling arm, not a KUKA digital twin. Dimensions, fixtures, tooling, and motion were chosen for this visualization.

## Second-pass dimensional references

- [KUKA KR QUANTEC range](https://www.kuka.com/en-de/products/robot-systems/industrial-robots/kr-quantec): industrial-arm size and reach context. The generic model's main links now sum to 3.1875 m; this is not equated to a manufacturer's rated reach or work envelope.
- [Schmalz SAB 100 NBR-60 RA dimensions](https://www.schmalz.com/en/vacuum-technology-for-automation/vacuum-components/vacuum-suction-cups/bellows-suction-cups-round/bellows-suction-cups-sab-1-5-folds-303189/10.01.06.01058/): a 100 mm-class bellows cup has a listed 100.8 mm suction diameter, 56.3 mm height, and recommended 6 mm hose inner diameter. These dimensions informed the generic model's 100 mm contact lip, 61 mm bellows, and 10 mm hose outside diameter. The model does not reproduce this product or inherit its ratings.
- The existing W203 photograph remains the visual reference for formed sunroof edges, console wells, perimeter returns, and smaller mounting openings. Final headliner dimensions (approximately 1.421 × 1.870 m, 0.165 m form depth) are design choices for a representative passenger-car roof, not dimensions extracted from that photograph.

All saved scenes use metric units at 1 m per Blender unit. Actual mesh measurements, hardware dimensions, and the distinction between geometric form depth and nominal material thickness are recorded in [dimensions.json](../validation/dimensions.json).
