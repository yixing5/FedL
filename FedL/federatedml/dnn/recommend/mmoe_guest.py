'''
Author: Sasha
Date: 2023-02-27 14:50:20
LastEditors: Sasha
LastEditTime: 2023-04-04 16:03:44
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend/mmoe_guest.py
'''
from federatedml.data_factory import multi_task_split_data
from FedL.federatedml.dnn.recommend.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_guest
from FedL.federatedml.PSI.psi_client import PSIClient
from FedL.conf.global_conf import ServiceConf
from FedL.util.log import log
from FedL.federatedml.util.secure_random import  generate_secure_noise
from FedL.federatedml.dnn.recommend.mmoe_bottom_model import MMoEBottomModel
from FedL.federatedml.dnn.recommend.mmoe_top_model import MMoETopModel
from FedL.federatedml.dnn.recommend.mmoe_interactive_layer import MMoEInteractiveModel
from FedL.federatedml.util.learn_rate_decay import CosineDecayWithWarmup 
from FedL.federatedml.util.early_stopping import EarlyStopping



class MMoEGuest(ModelBase):
    def __init__(self,train_x,label1,label2,feature_columns,feature_num_A,config) -> None:
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
        
        self.decay = CosineDecayWithWarmup(self.config['epoch'],self.config['lr_max'])
        self.bottom_model = MMoEBottomModel(feature_columns=self.feature_columns,embed_dims=self.config['embed_dims'],lr=self.config['bottom_conf']['lr'])
        self.top_model = MMoETopModel(experts_units=self.config['experts_units'],experts_num=self.config['experts_num'],lr=self.config['top_conf']['lr'])
        self.interactive_model = MMoEInteractiveModel(shape_host=(self.config["embed_dims"]*feature_num_A ,self.interactive_out_dim),
                                                     shape_guest=(self.config["embed_dims"]*len(self.feature_columns),self.interactive_out_dim),
                                                     lr = self.config['interactive_conf']['lr'],
                                                     decay = self.decay)
    
    def forward(self,input,n,name='train'):
        #####################################################获取bottom输出结果,计算交互层输出 ##############################################################
        
        output_bottom_guest = self.bottom_model(input).numpy()
        u_guest = self.interactive_model.forward_guest(output_bottom_guest,name=name)
        encrypted_output_bottom_host = self.get("encrypted_output_bottom_host_%s" % n)
        encrypted_u_host_sub_acc = self.interactive_model.forward_host(encrypted_output_bottom_host,name=name)
        
        mask = generate_secure_noise(encrypted_u_host_sub_acc.shape)
        encrypted_marked_u_host_sub_acc = encrypted_u_host_sub_acc +  mask
        self.remote(encrypted_marked_u_host_sub_acc,"encrypted_marked_u_host_sub_acc_%s" % n,ServiceConf.port_host,ServiceConf.ip_host)
        marked_u_host = self.get("marked_u_host_%s" % n)
        u_host = marked_u_host - mask
        output_interactice = u_host + u_guest
        return output_interactice
    
    def backward(self,input,label1,label2,n):
        #####################################################top model 训练 ,更新guest bottom_model##############################################################
        grads_act = self.top_model.train(input,label1,label2)
        grad_bottom_guest = self.interactive_model.get_grad_bottom_guest(grads_act)  
        self.bottom_model.train(self.X,grad_bottom_guest)
        #####################################################反向传播 计算交互层梯度 ###############################################################
        encryped_grad_weight_host = self.interactive_model.backward_host(grads_act) 
        
        mask_back = generate_secure_noise(encryped_grad_weight_host.shape)
        encryped_marked_grad_weight_host = encryped_grad_weight_host + mask_back
        self.remote(encryped_marked_grad_weight_host,"encryped_marked_grad_weight_host_%s" % n,ServiceConf.port_host,ServiceConf.ip_host)
        marked_grad_weight_host_sub_noise = self.get("marked_grad_weight_host_sub_noise_%s" % n)
        grad_weight_host_sub_noise = marked_grad_weight_host_sub_noise - mask_back
        #####################################################计算grad_bottom_host###########################################################
        encrypted_acc_noise = self.get("encrypted_acc_noise_%s" % n)
        encryped_grad_bottom_host = self.interactive_model.get_encryped_grad_bottom_host(grads_act,encrypted_acc_noise)
        self.remote(encryped_grad_bottom_host,"encryped_grad_bottom_host_%s" % n,ServiceConf.port_host,ServiceConf.ip_host)
        #####################################################更新交互层参数##########################################################
        grad_weight_guest = self.interactive_model.backward_guest(grads_act) 
        self.interactive_model.train(grad_weight_host_sub_noise ,grad_weight_guest)
        
    def run(self,n):
        output_interactice = self.forward(self.X,n,name='train') 
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
        feature_num_A = len(feature_columns_A)
        #隐私求交
        # ids = [x for x in range(len(XA_train)-10)]
        # psi_service = PSIService(ids = ids)
        # common_ids = psi_service.run()
        # XB_train ,y_train= XB_train[common_ids],y_train[common_ids]
        
        #训练
        mmoe_guest = MMoEGuest(XB_train,y1_train,y2_train,feature_columns_B,feature_num_A,config)
        early_stopping = EarlyStopping()
        for n in range(config['epoch']):
            mmoe_guest.run(n)
            if n % 50 == 0:
                valid_loss1,valid_loss2 = mmoe_guest.valid(XB_test,y1_test,y2_test)
                early_stopping(valid_loss1+valid_loss2)
                if early_stopping.early_stop:
                    log.info("Early stopping")
                    mmoe_guest.save()
                    break
                
        #预测
        # mmoe_guest.load_model()
        mmoe_guest.predict(XB_test)
    finally:
        mmoe_guest.fed_lmdb.drop()
    