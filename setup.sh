# (Optional) DCN is supported in CenterPoint, but not used in CenterFormer
# cd det3d/ops/dcn 
# python setup.py build_ext --inplace

# cd .. && cd  iou3d_nms
# python setup.py build_ext --inplace

cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace

cd ../.. && cd models/ops/
python setup.py build install
# unit test (should see all checking is True)
python test.py
