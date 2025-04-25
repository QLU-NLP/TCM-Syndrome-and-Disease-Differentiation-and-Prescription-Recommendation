export LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
python /app/src/task1/infer_diease.py
python /app/src/task1/infer_syndrome.py
python /app/src/task2/infer.py
python /app/src/hebing.py