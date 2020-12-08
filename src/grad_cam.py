   def make_grand_cam_heatmap(self,
                               image,
                               last_conv_layer_name,
                               classifier_layers_names):
        last_conv_layer = self.model_resnet.get_layer(last_conv_layer_name)
        inputs = self.model.inputs
        output = last_conv_layer.output
        last_conv_layer_model = Model(inputs,
                                      output)
        classifier_model = self.grand_cam_model(classifier_layers_names)
        grads, last_conv_output = self.calc_gradient(classifier_model,
                                                     last_conv_layer_model,
                                                     image)
        last_conv_layer_output = self.important_channel(last_conv_output,
                                                        grads)
        heatmap = self.create_heatmap(last_conv_layer_output)
        return heatmap

    def grand_cam_model(self, classifier_layer_names):
        input_shape = self.model_resnet.output.shape[1:]
        classifier_input = keras.Input(shape=input_shape)
        layer = classifier_input
        for layer_name in classifier_layer_names:
            layer = self.model.get_layer(layer_name)(layer)
        classifier_model = keras.Model(classifier_input, layer)
        return classifier_model

    def calc_gradient(self,
                      classifier_model,
                      last_conv_layer_model,
                      img):
        with GradientTape() as tape:
            last_conv_output = last_conv_layer_model(img)
            tape.watch(last_conv_output)
            preds = classifier_model(last_conv_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_chanel = preds[:, top_pred_index]
        grads = tape.gradient(top_class_chanel,
                              last_conv_output)
        return grads, last_conv_output

    def important_channel(self,
                          last_conv_layer_output,
                          grads):
        pooled_grads = tf.reduce_mean(grads, (0, 1, 2))
        last_conv_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_output[:, :, i] *= pooled_grads[i]
        return last_conv_output

    def create_heatmap(self, last_conv_output):
        heatmap = np.mean(last_conv_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap
