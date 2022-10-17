# to get the training loss and accuracy of full batch train and mini batch train  
`./run_figure4.sh`    
Full batch train 500 epoch will cost no more than an hour.                    
When mini batch size equals 16, it might cost a few hours on a single gpu.  
(each mini batch training time is short, but each epoch will last long. each epoch will have about 1223 mini batches)

`cd draw_figure/` 
`python draw_figure.py`    
based on the training logs of full batch train and mini batch train   
to generate Figure4 (full_v.s._mini.png)  
To save time, we use the training log already collected to draw the figure.  