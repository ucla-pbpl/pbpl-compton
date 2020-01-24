set -x
echo "start running"
for i in {1..5}
do
	echo "##### run for test ${i} #####"
    bash run.sh
    mv *.npz out
    mv out/* ../train/test-right-y
done
for i in {1..15}
do
	echo "##### run for train ${i} #####"
    bash run.sh
    mv *.npz out
    mv out/* ../train/train-right-y
done
#cd ../train
#python preprocess_image.py
echo "done"
