
#!/bin/bash  
echo Running all 	 

problem=3
replica=10
swapint=2
samples=100
maxtemp=3
burn=0.25
pt_stage=0.5
raintimeint=4
initialtopoep=0.5

echo $problem 


 
for t in 1 #4 8 16
	do  
 
		 	python visualise.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint

  
  
	done 

