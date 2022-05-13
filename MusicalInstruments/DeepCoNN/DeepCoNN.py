class DeepCoNN(object):
    def __init__(
            self, user_length,item_length, num_classes, user_vocab_size,item_vocab_size,fm_k,n_latent,user_num,item_num,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,l2_reg_V=0.0):
        self.input_u = tf.compat.v1.placeholder(tf.int32, [None, user_length], name="input_u")
        self.input_i = tf.compat.v1.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None,1],name="input_y")
        self.input_uid = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.compat.v1.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random.uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_user = tf.nn.embedding_lookup(params=self.W1, ids=self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        with tf.compat.v1.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random.uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_item = tf.nn.embedding_lookup(params=self.W2, ids=self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        pooled_outputs_u = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.compat.v1.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    input=self.embedded_users,
                    filters=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool2d(
                    input=h,
                    ksize=[1, user_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_u.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_u = tf.concat(3,pooled_outputs_u)
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, num_filters_total])

        pooled_outputs_i = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.compat.v1.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    input=self.embedded_items,
                    filters=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool2d(
                    input=h,
                    ksize=[1, item_length- filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(3,pooled_outputs_i)
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, num_filters_total])

        with tf.compat.v1.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, rate=1 - (1.0))
            self.h_drop_i= tf.nn.dropout(self.h_pool_flat_i, rate=1 - (1.0))
        with tf.compat.v1.name_scope("get_fea"):
            Wu = tf.compat.v1.get_variable(
                "Wu",
                shape=[num_filters_total, n_latent],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_fea=tf.matmul(self.h_drop_u, Wu) + bu
            #self.u_fea = tf.nn.dropout(self.u_fea,self.dropout_keep_prob)
            Wi = tf.compat.v1.get_variable(
                "Wi",
                shape=[num_filters_total, n_latent],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_fea = tf.matmul(self.h_drop_i, Wi) + bi
            #self.i_fea=tf.nn.dropout(self.i_fea,self.dropout_keep_prob)

         
        with tf.compat.v1.name_scope('fm'):
            self.z=tf.nn.relu(tf.concat(1,[self.u_fea,self.i_fea]))

            #self.z=tf.nn.dropout(self.z,self.dropout_keep_prob)

            WF1=tf.Variable(
                    tf.random.uniform([n_latent*2, 1], -0.1, 0.1), name='fm1')
            Wf2=tf.Variable(
                tf.random.uniform([n_latent*2, fm_k], -0.1, 0.1), name='fm2')
            one=tf.matmul(self.z,WF1)

            inte1=tf.matmul(self.z,Wf2)
            inte2=tf.matmul(tf.square(self.z),tf.square(Wf2))
  
            inter=(tf.square(inte1)-inte2)*0.5

            inter=tf.nn.dropout(inter,rate=1 - (self.dropout_keep_prob))

            inter=tf.reduce_sum(input_tensor=inter,axis=1,keepdims=True)
            print(inter)
            b=tf.Variable(tf.constant(0.1), name='bias')
            

            self.predictions =one+inter+b

            print(self.predictions)
        with tf.compat.v1.name_scope("loss"):
            #losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.compat.v1.name_scope("accuracy"):
            self.mae = tf.reduce_mean(input_tensor=tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy =tf.sqrt(tf.reduce_mean(input_tensor=tf.square(tf.subtract(self.predictions, self.input_y))))