from model import *
from data import *
from sklearn.model_selection import train_test_split
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

# myGene = trainGenerator(5,r'D:\rubik talent\segmentation\unet-master\Train_gray','image_gray','label_gray',aug_dict=data_gen_args,save_to_dir = None)
# myVal=trainGenerator(5,r'D:\rubik talent\segmentation\unet-master\Val_gray','image_gray','label_gray',aug_dict=data_gen_args,save_to_dir = None)

# model = unet(r'D:\rubik talent\segmentation\best_model.hdf5')
# model_checkpoint = ModelCheckpoint(r'D:\rubik talent\segmentation\test.hdf5', monitor='loss',verbose=1, save_best_only=True)
# history=model.fit_generator(myGene,steps_per_epoch=4000,validation_data=myVal,validation_steps=200,epochs=50,callbacks=[model_checkpoint])
# Load the best model
model = tf.keras.models.load_model(r'D:\rubik talent\segmentation\best_model.hdf5')
testGene = testGenerator(r"D:\rubik talent\segmentation\unet-master\Predict")
results = model.predict_generator(testGene,10,verbose=1)
saveResult(r"D:\rubik talent\segmentation\unet-master\Predict",results)