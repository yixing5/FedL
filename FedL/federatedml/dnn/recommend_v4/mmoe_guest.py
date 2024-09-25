'''
Author: Sasha
Date: 2023-02-27 14:50:20
LastEditors: Sasha
LastEditTime: 2023-04-14 09:20:04
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend_v4/mmoe_guest.py
'''
from federatedml.data_factory import multi_task_split_data
from FedL.federatedml.dnn.recommend_v4.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_guest
from FedL.federatedml.PSI.psi_client import PSIClient
from FedL.conf.global_conf import ServiceConf
from FedL.util.log import log
from FedL.federatedml.util.secure_random import  generate_secure_noise
from FedL.federatedml.dnn.recommend_v4.mmoe_bottom_model import MMoEBottomModel
from FedL.federatedml.dnn.recommend_v4.mmoe_top_model import MMoETopModel
from FedL.federatedml.util.learn_rate_decay import CosineDecayWithWarmup 
from FedL.federatedml.util.early_stopping import EarlyStopping
import numpy as np


class MMoEGuest(ModelBase):
    def __init__(self,train_x,label1,label2,feature_columns,config) -> None:
        super().__init__(fed_lmdb=feddb_guest)
        self.config = config
        self.X = train_x
        self.label1 = label1
        self.label2 = label2
        self.bottom_model_name = "mmoe_guest_bottom.h5"
        self.top_model_name = "mmoe_guest_top.h5"
        self.interactive_model_name = "mmoe_guest_interactive.h5"
        self.feature_columns = feature_columns
        self.interactive_out_dim = self.config['task_num']*self.config['experts_num'] + self.config['experts_num']*self.config['experts_units']
        self.acc_noise = None
        self.decay = CosineDecayWithWarmup(self.config['epoch'],self.config['lr_max'])
        self.bottom_model = MMoEBottomModel(feature_columns=self.feature_columns,embed_dims=self.config['embed_dims'],lr=self.config['bottom_conf']['lr'])
        self.top_model = MMoETopModel(experts_units=self.config['experts_units'],experts_num=self.config['experts_num'],lr=self.config['top_conf']['lr'])
        self.weight_acc = None
        self._generate_key()
    
    def forward(self,input,n):
        #####################################################获取bottom输出结果,计算交互层输出 ##############################################################
        
        self.output_bottom_guest = self.bottom_model(input).numpy()
        encrypted_output_bottom_guest = self.public_key.encrypt_list(self.output_bottom_guest)
        self.remote(encrypted_output_bottom_guest,"encrypted_output_bottom_guest_%s" % n,ServiceConf.port_host,ServiceConf.ip_host)
        
        # marked_u_host_sub_acc = self.private_key.decrypt_list(encrypted_marked_u_host_sub_acc)
        if self.acc_noise is None:
            self.acc_noise = np.zeros((self.output_bottom_guest.shape[1], self.interactive_out_dim))
            
        # marked_u_host = marked_u_host_sub_acc + output_bottom_host @ self.acc_noise
        # self.remote(marked_u_host,"marked_u_host_%s" % n,ServiceConf.port_host,ServiceConf.ip_host)
        encrypted_output_interactice = self.get("encrypted_output_interactice_%s" % n)
        output_interactice_with_noise = self.private_key.decrypt_list(encrypted_output_interactice)
        
        output_interactice = output_interactice_with_noise + self.output_bottom_guest @ self.acc_noise
        return output_interactice
    
    def backward(self,input,label1,label2,n):
        #####################################################top model 训练 ,更新guest bottom_model##############################################################
        ###step 2
        grads_act = self.top_model.train(input,label1,label2)
        encryped_acc_noise = self.public_key.encrypt_list(self.acc_noise)
        tmp_noise = generate_secure_noise((self.output_bottom_guest.shape[1], self.interactive_out_dim))

        encryped_grad_weight_guest = self.output_bottom_guest.T @ grads_act / len(grads_act)
        delta_grad_weight_guest = encryped_grad_weight_guest  + tmp_noise/self.decay.compute_step()
        
        self.remote(grads_act,"grads_act_%s" % n,ServiceConf.port_host,ServiceConf.ip_host)
        self.remote(encryped_acc_noise,"encryped_acc_noise_%s" % n,ServiceConf.port_host,ServiceConf.ip_host)
        self.remote(delta_grad_weight_guest,"delta_grad_weight_guest_%s" % n,ServiceConf.port_host,ServiceConf.ip_host)

        
         ###step 5
        encryped_bottom_guest =  self.get("encryped_grad_bottom_guest_%s" % n)
        self.acc_noise += tmp_noise
        self.bottom_model.train(self.X,encryped_bottom_guest)

        
    def run(self,n):
        output_interactice = self.forward(self.X,n) 
        self.backward(output_interactice,self.label1,self.label2,n)       
        
    def predict(self,X_test):
        output_interactice = self.forward(X_test,'predict',name='predict')
        pred = self.top_model(output_interactice)
        self.remote(pred,"pred",ServiceConf.port_host,ServiceConf.ip_host)
        log.info('pred: %s '% pred)
    
    def valid(self,inputs,label1,label2,n='valid'):
        output_interactice = self.forward(inputs,n,name='valid')
        predictions_task1,predictions_task2,valid_loss1,valid_loss2,valid_auc1,valid_auc2 = self.top_model.valid(output_interactice,label1,label2)
        self.remote([valid_loss1,valid_loss2],"valid_loss",ServiceConf.port_host,ServiceConf.ip_host)
        log.info('######valid_loss: %s,%s,%s########   auc1:  %s########### auc2:  %s#######' % (valid_loss1.numpy(),valid_loss2.numpy(),(valid_loss1+valid_loss2).numpy(),valid_auc1.numpy(),valid_auc2.numpy()))
        return valid_loss1,valid_loss2

    def save(self):
        self.bottom_model.save(self.bottom_model_name)
        self.top_model.save(self.top_model_name)
        self.interactive_model.save(self.interactive_model_name)
    
    def load_model(self):
        self.bottom_model.load_model(self.bottom_model_name)
        self.top_model.load_model(self.top_model_name)
        self.interactive_model.load_model(self.interactive_model_name)
            
            
if __name__ == "__main__":
    try:
        feature_columns_A,feature_columns_B,XA_train,XB_train,XA_test,XB_test, y1_train, y1_test,y2_train, y2_test = multi_task_split_data()
        #隐私求交
        # ids = [x for x in range(len(XA_train)-10)]
        # psi_service = PSIService(ids = ids)
        # common_ids = psi_service.run()
        # XB_train ,y_train= XB_train[common_ids],y_train[common_ids]
        
        #训练
        mmoe_guest = MMoEGuest(XB_train,y1_train,y2_train,feature_columns_B,config)
        early_stopping = EarlyStopping()
        for n in range(config['epoch']):
            mmoe_guest.run(n)
            # if n % 50 == 0:
            #     valid_loss1,valid_loss2 = mmoe_guest.valid(XB_test,y1_test,y2_test)
            #     early_stopping(valid_loss1+valid_loss2)
            #     if early_stopping.early_stop:
            #         log.info("Early stopping")
            #         mmoe_guest.save()
            #         break
                
        #预测
        # mmoe_guest.load_model()
        mmoe_guest.predict(XB_test)
    finally:
        mmoe_guest.fed_lmdb.drop()
    