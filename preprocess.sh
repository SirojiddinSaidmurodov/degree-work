#!/bin/bash
echo "Starting preprocessing"
if [ -d output ]
then
    rm -r output
    mkdir output
else
    mkdir output
fi
cnt=0
for i in ./*.doc
do
    cnt=$((cnt+1))
done
total="$cnt"

for file in ./*.doc
do
if [ -f "$file" ]
then
    echo "$file"
    touch ./output/$file.txt 
    tika --text $file > ./output/$file.txt 2>/dev/null
fi
done | tqdm --total $total >> /dev/null
echo "Done!"