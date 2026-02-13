mkdir -p ./data
/usr/bin/time -l python main.py 1> ./data/stdout.log 2> ./data/memory.dat
