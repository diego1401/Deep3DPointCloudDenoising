# Deep3DPointCloudDenoising

![alt text](extra/init_figure.png?raw=true "Title")

This is a repository of a U-Net based method to clean 3D point clouds.
In here you can find different files that I used to implement, train and test the models.

To use it start by building the docker image using the [Dockerfile](https://docs.docker.com/engine/reference/commandline/build/).
Then every action onwards is done in the docker image:
- Run the file compile_ops.sh
- Add a folder with subfolders training, validation and testing with the corresponding shapes. We used the [PointCleanNet Dataset](https://github.com/mrakotosaon/pointcleannet).
- To train the model for the Offset Regression task use the train_dist.py file and the l1.yaml config file.

See the extra folder for the report and presentation that I used to defend this internship!

