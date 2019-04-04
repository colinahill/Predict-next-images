### Predicts the next image in a sequence using a convolutional LSTM network
### Basic example using moving squares

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import pylab as pl
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def generate_data(nmovies, nframes, n):
### Function to generate multiple movies with moving squares in the frames
### frames are nxn pixels in size, and moving squares are randomly sized between 3-6 pixels across
### Squares move linearly over time
### INPUT : nmovies = number of movies to generate
###         nframes = number of frames to generate
###         n = length of side of image (number of pixels)
    movies          = np.zeros((nmovies, nframes, n*3, n*3, 1), dtype=np.float)
    movies_shifted  = np.zeros((nmovies, nframes, n*3, n*3, 1), dtype=np.float)

### Loop through movies
    for i in range(nmovies):

        ### generate nsquares
        nsquares = np.random.randint(3,7)
        for j in range(nsquares):

        ### initialize starting point for square
            x_start = np.random.randint(n,2*n)
            y_start = np.random.randint(n,2*n)
        ### initialize direction of motion
            x_direction = np.random.randint(0,3) - 1 ### move left, don't move, or move right
            y_direction = np.random.randint(0,3) - 1 ### move left, don't move, or move right

            ### size of square
            m = np.random.randint(3,7)

            ### Move square across image
            for k in range(nframes):
                x_shift = x_start + x_direction * k
                y_shift = y_start + y_direction * k
                ### squares origin is bottom-left
                movies[i, k, x_shift:x_shift+m, y_shift:y_shift+m, 0] += 1

                ### Add noise around square
                if(np.random.randint(0,2)): ### Add noise randomly to frames
                    movies[i, k, x_shift-1:x_shift+m+1, y_shift-1:y_shift+m+1, 0] += 0.1 * (-1)**np.random.randint(0, 2) ### Add or subtract 0.1 to surrounding pixels

                ### Make a shifted version of the frames 1 time step ahead
                x_shift = x_start + x_direction * (k + 1)
                y_shift = y_start + y_direction * (k + 1)
                movies_shifted[i, k, x_shift:x_shift+m, y_shift:y_shift+m, 0] += 1

    movies = movies[:, :, n:2*n, n:2*n, :]
    movies_shifted = movies_shifted[:, :, n:2*n, n:2*n, :]
    movies[movies >=1] = 1
    movies_shifted[movies_shifted >=1] = 1
    return movies, movies_shifted

if __name__ == '__main__':

    ### Reproducibility of results 
    np.random.seed(123)

    nmovies = 1000 ### Number of frames in movie
    nframes = 15 ### Number of frames in movie
    n = 32 ### Number of pixels in image (nxn)
    movies, movies_shifted = generate_data(nmovies, nframes, n)

### Plot examples from test data
    fig, ax = pl.subplots(1,2)
    for i in range(3):
        for j in range(nframes):
            ax[0].imshow(movies[i, j, :, :, 0])
            ax[1].imshow(movies_shifted[i, j, :, :, 0])
            pl.savefig("im_{:03d}_{:03d}".format(i,j),vmin=-0.1,vmax=1.1)

    ### Create model:
    ### first layer has input shape of (nmovies, width, height, channels)
    ### returns identical shape
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),input_shape=(None, n, n, 1), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),input_shape=(None, n, n, 1), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),input_shape=(None, n, n, 1), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),input_shape=(None, n, n, 1), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta')


    ### Split data into training and test sets
    trainingfraction = 0.9
    train_size = round(nmovies * trainingfraction)

    ### Train the network
    ### validate_split is the fraction of the data to use in calculating the loss
    early_stopping_monitor = EarlyStopping(patience=3)
    model.fit(movies[:train_size], movies_shifted[:train_size], batch_size=10, epochs=100, validation_split=0.05,callbacks=[early_stopping_monitor])


    ### Take an example from the test set and predict the next steps
    index = train_size+1

    num_test_frames = 7 ### Number of frames to predict

    train_pred = movies[index][:nframes-num_test_frames,:,:,:]
    for j in range(nframes):
        new_pos = model.predict(train_pred[np.newaxis, :, :, :, :])
        new = new_pos[:, -1, :, :, :]
        train_pred = np.concatenate((train_pred, new), axis=0)

    ### Compare predictions to the truth
    truth = movies[index][:, :, :, :]

    vmin = 0
    vmax = 1.0
    for i in range(nframes):
        fig, ax = pl.subplots(1,2)
        ### In left panel show original then predicted frames
        if i >= (nframes-num_test_frames):
            ax[0].set_title('Prediction')
        else:
            ax[0].set_title('Original')
        ax[0].imshow(train_pred[i, :, :, 0],vmin=vmin,vmax=vmax)

        ### In right panel show only truth 
        ax[1].set_title('truth')
        if i >= 2:
            ax[1].imshow(movies_shifted[index][i - 1, :, :, 0],vmin=vmin,vmax=vmax)
        else:
            ax[1].imshow(truth[i, :, :, 0],vmin=vmin,vmax=vmax)
        pl.savefig("im_{:03d}.png".format(i + 1))