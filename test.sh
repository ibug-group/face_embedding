#!/bin/bash

# iresnet18 backbone
python face_embedding_test.py --input1 "samples/Niki_DeLoach_1.png" \
--input2 "samples/Peri_Gilpin_1.png" -b iresnet18

# iresnet18 backbone, project to roi_tanh_polar space, and do not align face
python face_embedding_test.py --input1 "samples/Niki_DeLoach_2.png" \
--input2 "samples/Stratos_Dionisiou_1.png" -b iresnet18 --align_face 0 \
--project_to_space roi_tanh_polar

# iresnet50 backbone, project to roi_tanh space, and do not flip face
python face_embedding_test.py --input1 "samples/Peri_Gilpin_1.png" \
--input2 "samples/Peri_Gilpin_2.png" -b iresnet18 --flip 0 \
--project_to_space roi_tanh

## iresnet50 backbone, project to roi_tanh space, and do not flip face
#python face_embedding_test.py --input1 "samples/Peri_Gilpin_1.png" \
#--input2 "samples/Peri_Gilpin_2.png" -b iresnet50 --flip 0 \
#--project_to_space roi_tanh
#
## rtnet50 (project_to_space should be set to roi_tanh_polar)
#python face_embedding_test.py --input1 "samples/Stratos_Dionisiou_1.png" \
#--input2 "samples/Stratos_Dionisiou_2.png" -b rtnet50 --project_to_space roi_tanh_polar
