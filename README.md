The Trojan'd model is in src/model.py

to maximally parallelize the results you need to increase the number of files your OS is will to have open at a given time.
```ulimit -n 10000```
That command should work on linux.
