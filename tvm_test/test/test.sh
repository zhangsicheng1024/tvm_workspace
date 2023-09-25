dim_grid=$(sed -n "1, 1p" ./t.txt | awk -F' ' '{print $1}')
dim_blk=$(sed -n "2, 1p" ./t.txt | awk -F' ' '{print $1}')
echo $dim_grid
echo $dim_blk
sed -i "13idim3 dimGrid($dim_grid, 1, 1);\ndim3 dimBlock($dim_blk, 1, 1);\n" ./t.cu