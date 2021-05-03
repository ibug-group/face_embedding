#!/bin/bash

# iresnet18 backbone, original (Cartesian) space
python face_embedding_test.py --input1 "samples/Niki_DeLoach_1.png" \
--input2 "samples/Peri_Gilpin_1.png" -b iresnet18

# iresnet18 backbone, project to RoI Tanh Polar space, and do not align faces
python face_embedding_test.py --input1 "samples/Niki_DeLoach_2.png" \
--input2 "samples/Stratos_Dionisiou_1.png" -b iresnet18 --align_face 0 \
--project_to_space roi_tanh_polar

# iresnet50 backbone, project to RoI Tanh space, and do not flip faces
python face_embedding_test.py --input1 "samples/Peri_Gilpin_1.png" \
--input2 "samples/Peri_Gilpin_2.png" -b iresnet50 --flip 0 \
--project_to_space roi_tanh

# iresnet50 backbone, project to RoI Tanh Circular space, do not flip faces, and do not align faces
python face_embedding_test.py --input1 "samples/Stratos_Dionisiou_2.png" \
--input2 "samples/Peri_Gilpin_2.png" -b iresnet50 --flip 0 --align_face 0 \
--project_to_space roi_tanh_circular


# rtnet50 (project_to_space should be set to roi_tanh_polar)
python face_embedding_test.py --input1 "samples/Stratos_Dionisiou_1.png" \
--input2 "samples/Stratos_Dionisiou_2.png" -b rtnet50 --project_to_space roi_tanh_polar
