./run_figure4.sh    # to get the training loss and accuracy
                    # when mini batch size equals 16, each mini batch training time is short, but each epoch will last long.
cd draw_figure/
python draw_figure.py   
                    # based on the training logs of full batch train and mini batch train   
                    # to generate Figure4 (full_v.s._mini.png)
                    # To save time, we use the training log already collected to draw the figure.