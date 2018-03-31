class GAN():
    
    def init_v17_v10_plot(self):
        '''Functions to initalise figure/plot variables for specific features, 
        by setting up figure and plotting real fraud data on the left hand side for comparisons. '''
    def init_v17_v14_plot(self):

    def load_fraud_data(self):
        ''' Read in the dataset and do all of the necessary preprocessing (Feature scaling etc) 
        and store in accessible class variables.'''
        
    def __init__(self):
        '''Calls any initalisation functions, builds generator and discriminator networks, 
        compile the networks, set up combined model.'''
 
    def build_generator(self):
        '''Uses Keras functional API to create the generator network.'''

    def build_discriminator(self):
        '''Uses Keras functional API to create the discriminator network.'''
    
    def train(self, epochs, batch_size=128, save_interval=200):
        '''The adversarial training function for the GAN.'''
        
    def save_loss_plot(self):
        '''Utility to save the plot of losses for the GAN.'''
        
    def save_imgs(self, epoch, img, gen_imgs): 
        '''Utility to save the plot of generated output for specified save intervals, 
        during the training process.'''        
    
    def test_as_classifier(self):
        '''For use in GAN-SSL. Wraps up all the evaluation procedures for predictions and retrieving results metrics.'''

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=6000, batch_size=32, save_interval=1000)
    gan.test_as_classifier()