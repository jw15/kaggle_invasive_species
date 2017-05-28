'''
Initiate EC2 instance with GPU.
This time: 'hyacinth_gpu_retry' from dsi-template (because last time I messed up the dsi template by installing newer versions of theano, etc.). Could also start from scratch but that seems REALLY time consuming.)
Set elastic IP address and update config file with instance name, address.
ssh into EC2 instance and initiate tmux session:
'''
# from inside .ssh folder:
ssh hyacinth_gpu_retry

# from inside ubuntu server window that's connected to EC2 instance:
tmux new -s hyacinth_retry

# Copy files to EC2 instance (from terminal window inside .ssh folder)
#copy updated bashrc to use Theano, not tensorflow backend
scp -i demo.pem ~/Downloads/.bashrc ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu/.bashrc

# copy .theanorc file to hopefully set up GPU use as default.
scp -i demo.pem ~/.theanorc ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu

# copy file to test if theano IS using the gpu:
scp -i demo.pem ~/kaggle/invasive/test_theano.py ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu


# update stuff on the EC2 instance:
conda update conda

# Update keras
pip install keras --upgrade


# Ensure that keras is using theano backend:
scp -i demo.pem ~/.keras/keras.json ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu/.keras/keras.json



# access files on server on local machine:
# make dir for files:
mkdir mnt
cd mnt
mkdir droplet

# none of the following work:
sudo sshfs -o allow_other,defer_permissions ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/ mnt/droplet

sudo sshfs -o allow_other,defer_permissions,IdentityFile=~/.ssh/demo.pem ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/ mnt/droplet

sshfs ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu mnt/droplet


'''
Compress train and test folders (images)
'''

# update anaconda:
conda update --prefix /home/ubuntu/anaconda3 anaconda

# install cv2
pip install opencv-python

# install unzip
sudo apt install unzip

# unzip numpy array
unzip 224.npz

# transfer .bashrc file over to set Theano as backend.
# scp -i demo.pem ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu/.bashrc ~/Downloads/

scp -i demo.pem ~/Downloads/.bashrc ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu/.bashrc

scp -i demo.pem ~/kaggle/invasive/example_cnn_kaggle_onlycnn.py ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu/invasive

scp -i demo.pem -r ~/kaggle/invasive/224.npz.zip ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu/

scp -i demo.pem ~/kaggle/invasive/data/sample_submission.csv ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu/invasive/

scp -i demo.pem ~/kaggle/invasive/data/train_labels.csv ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu/invasive/

# get submission csv entry from EC2:

scp -i demo.pem ubuntu@ec2-34-225-25-74.compute-1.amazonaws.com:/home/ubuntu/invasive/submit.csv ~/kaggle/invasive/
