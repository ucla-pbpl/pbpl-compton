set -x
echo "start running"
for i in {1..10}
do
	echo "##### run for test ${i} #####"
    bash run.sh
    mv *.npz out
    mv out/* ../train/test
done
for i in {1..30}
do
	echo "##### run for train ${i} #####"
    bash run.sh
    mv *.npz out
    mv out/* ../train/train
done
#cd ../train
#python preprocess_image.py
echo "done"
