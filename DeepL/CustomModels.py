import tensorflow as tf
from tensorflow.keras import models, layers, metrics

class DotProd(layers.Layer):
    def __init__(self, units, *args, **kwargs):
        super(DotProd, self).__init__(*args, **kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        b, phi = x
        #tf.einsum("bi,ni->bn", b, phi) + self.bias
        return tf.matmul(phi, tf.transpose(b)) + self.bias
    
class pod_deeponet(models.Model):
    def __init__(self, branch, *args, **kwargs):
        super(pod_deeponet, self).__init__(*args, **kwargs)
        self.branch = branch
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
        
        self.dotprod = DotProd(1)
    
    def compile_grid_modes(self, grid, modes):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
        self.modes = tf.convert_to_tensor(modes, dtype=tf.float32)
                
    
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        phi = self.modes
        
        u = self.dotprod([b, phi])
        return u
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u = self(inputs, training=True)
            loss = self.loss(outputs, u)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u = self(inputs, training=False)
        loss = self.loss(outputs, u)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    

class deeponet(models.Model):
    def __init__(self, branch, trunk, *args, **kwargs):
        super(deeponet, self).__init__(*args, **kwargs)
        self.branch = branch
        self.trunk = trunk
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
        
        self.dotprod_1 = DotProd(1)
    
    def compile_grid_modes(self, grid, modes = None):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        phi_ux, phi_uy, phi_uz = self.trunk(self.x)
        phi = tf.concat((phi_ux, phi_uy, phi_uz), axis=0)
        
        u = self.dotprod_1([b, phi])
        return u
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u = self(inputs, training=True)
            loss = self.loss(outputs, u)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u = self(inputs, training=False)
        loss = self.loss(outputs, u)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    

class pod_deepnet(models.Model):
    def __init__(self, branch, *args, **kwargs):
        super(pod_deepnet, self).__init__(*args, **kwargs)
        self.branch = branch
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
            
    def compile_grid_modes(self, grid, modes = None):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
        self.modes = tf.convert_to_tensor(modes, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        phi = self.modes
        
        u = phi @ tf.transpose(b)
        return tf.transpose(u), b
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        b_outputs = tf.transpose(tf.linalg.lstsq(self.modes, tf.transpose(outputs)))
        with tf.GradientTape() as tape:
            u, b = self(inputs, training=True)
            loss = self.loss(b_outputs, b)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        b_outputs = tf.transpose(tf.linalg.lstsq(self.modes, tf.transpose(outputs)))
        u, b = self(inputs, training=False)
        loss = self.loss(b_outputs, b)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
    
class deim_pod_deepnet(models.Model):
    def __init__(self, branch, *args, **kwargs):
        super(deim_pod_deepnet, self).__init__(*args, **kwargs)
        self.branch = branch
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
            
    def compile_grid_modes(self, grid, modes = None, PT = None):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
        self.modes = tf.convert_to_tensor(modes, dtype=tf.float32)
        self.PT = tf.convert_to_tensor(PT, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        phi = self.modes
        
        u = phi @ tf.transpose(b)
        return tf.transpose(u), b
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        b_outputs = tf.transpose(self.PT @ tf.transpose(outputs))
        with tf.GradientTape() as tape:
            u, b = self(inputs, training=True)
            loss = self.loss(b_outputs, b)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        b_outputs = tf.transpose(self.PT @ tf.transpose(outputs))
        u, b = self(inputs, training=False)
        loss = self.loss(b_outputs, b)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
    
class deim_deeponet(models.Model):
    def __init__(self, branch, trunk, *args, **kwargs):
        super(deim_deeponet, self).__init__(*args, **kwargs)
        self.branch = branch
        self.trunk = trunk
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
        
        self.dotprod_1 = DotProd(1)
    
    def compile_grid_modes(self, grid, modes = None, PT = None):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
        self.modes = tf.convert_to_tensor(modes, dtype=tf.float32)
        self.PT = tf.convert_to_tensor(PT, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        x = self.PT @ tf.concat([self.x, self.x], 0)
        phi = self.trunk(x)
        
        u = self.dotprod_1([b, phi])
        u = self.modes @ tf.transpose(u)
        
        return tf.transpose(u)
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u = self(inputs, training=True)
            loss = self.loss(outputs, u)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u = self(inputs, training=False)
        loss = self.loss(outputs, u)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
class deim_deeponet_r(models.Model):
    def __init__(self, branch, trunk, *args, **kwargs):
        super(deim_deeponet_r, self).__init__(*args, **kwargs)
        self.branch = branch
        self.trunk = trunk
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
        
        self.dotprod_1 = DotProd(1)
    
    def compile_grid_modes(self, grid, modes = None, PT = None):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
        self.modes = tf.convert_to_tensor(modes, dtype=tf.float32)
        self.PT = tf.convert_to_tensor(PT, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        x = self.PT @ tf.concat([self.x, self.x], 0)
        phi = self.trunk(x)
        
        u_r = self.dotprod_1([b, phi])
        u = self.modes @ tf.transpose(u_r)
        
        return tf.transpose(u), u_r
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        outputs_r = tf.transpose(self.PT @ tf.transpose(outputs))
        with tf.GradientTape() as tape:
            u, u_r = self(inputs, training=True)
            loss = self.loss(outputs_r, u_r)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        outputs_r = tf.transpose(self.PT @ tf.transpose(outputs))
        u, u_r = self(inputs, training=False)
        loss = self.loss(outputs_r, u_r)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
    
class diff_deim_deeponet_r(models.Model):
    def __init__(self, branch, trunk, *args, **kwargs):
        super(diff_deim_deeponet_r, self).__init__(*args, **kwargs)
        self.branch = branch
        self.trunk = trunk
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
        
        self.dotprod_1 = DotProd(1)
    
    def compile_grid_modes(self, grid, modes = None, PT = None):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
        self.modes = tf.convert_to_tensor(modes, dtype=tf.float32)
        self.PT = tf.convert_to_tensor(PT, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        inputs, x = inputs
        b = self.branch(inputs)
        x = self.PT @ x
        phi = self.trunk(x)
        
        u_r = self.dotprod_1([b, phi])
        u = self.modes @ tf.transpose(u_r)
        
        return tf.transpose(u), u_r
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        x = tf.concat([self.x, self.x], 0)
        outputs_r = tf.transpose(self.PT @ tf.transpose(outputs))
        with tf.GradientTape(persistent=True) as tape:
            u, u_r = self([inputs, x], training=True)
            loss = self.loss(outputs_r, u_r)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
    # @tf.function
    def diff_step(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = tf.concat([self.x, self.x], 0)
        xx = x[:, 0:1]
        yy = x[:, 1:]
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(xx)
            tape.watch(yy)
            x = tf.concat([xx, yy], axis = 1)
            u, u_r = self([inputs, x], training=True)
            u = tf.transpose(u)

        ux = tape.batch_jacobian(u, xx)
        uy = tape.batch_jacobian(u, yy)
        return ux, uy
    
    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        x = tf.concat([self.x, self.x], 0)
        outputs_r = tf.transpose(self.PT @ tf.transpose(outputs))
        u, u_r = self([inputs, x], training=False)
        loss = self.loss(outputs_r, u_r)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1)
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
    # @tf.function
    def predict(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = tf.concat([self.x, self.x], 0)
        u, u_r = self([inputs, x], training=False)
        return u.numpy()