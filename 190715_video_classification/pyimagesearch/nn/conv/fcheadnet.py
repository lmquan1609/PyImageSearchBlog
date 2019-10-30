from keras.layers import AveragePooling2D, Flatten, Dense, Dropout

class FCHeadNet:
    @staticmethod
    def build(base_model, D, classes, average_pool_size=(7, 7)):
        head_model = base_model.output
        head_model = AveragePooling2D(pool_size=average_pool_size)(head_model)
        head_model = Flatten(name='flatten')(head_model)
        head_model = Dense(D, activation='relu')(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(classes, activation='softmax')(head_model)
        return head_model