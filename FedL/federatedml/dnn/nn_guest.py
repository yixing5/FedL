from FedL.federatedml.data_factory import split_data
from FedL.federatedml.dnn.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_guest
from FedL.federatedml.PSI.psi_client import PSIClient
from FedL.conf.global_conf import ServiceConf
from FedL.util.log import log
from FedL.federatedml.util.secure_random import  generate_secure_noise
from FedL.federatedml.dnn.model.nn_bottom_model import BottomModel
from FedL.federatedml.dnn.model.nn_top_model import TopModel
from FedL.federatedml.dnn.model.interactive_layer import InteractiveModel
from FedL.federatedml.util.learn_rate_decay import CosineDecayWithWarmup 
from FedL.federatedml.util.early_stopping import EarlyStopping


class LinrGuest(ModelBase):
    def __init__(self,train_x,label,config) -> None:
        super().__init__(fed_lmdb=feddb_guest)
        self.config = config
        self.X = train_x
        self.y = label
        self.bottom_model_name = "nn_guest_bottom.h5"
        self.top_model_name = "nn_guest_top.h5"
        self.interactive_model_name = "nn_guest_interactive.h5"
        
        self.decay = CosineDecayWithWarmup(self.config['epoch'],self.config['lr_max'])
        self.bottom_model = BottomModel(self.config['bottom_conf']['layers_dims'],input_dim=self.X.shape[1],lr=self.config['bottom_conf']['lr'])
        self.top_model = TopModel(self.config['top_conf']['layers_dims'],input_dim=self.config['interactive_conf']['layer_dims'],lr=self.config['top_conf']['lr'])
        self.interactive_model = InteractiveModel(shape_host=(self.config['bottom_conf']['layers_dims'][-1],self.config['interactive_conf']['layer_dims']),
                                                 shape_guest=(self.config['bottom_conf']['layers_dims'][-1],self.config['interactive_conf']['layer_dims']),
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
    
    def backward(self,input,labels,n):
        #####################################################top model 训练 ,更新guest bottom_model##############################################################
        grads_act = self.top_model.train(input,labels)
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
        self.backward(output_interactice,self.y,n)       
        
    def predict(self,X_test):
        output_interactice = self.forward(X_test,'predict',name='predict')
        pred = self.top_model(output_interactice)
        self.remote(pred,"pred",ServiceConf.port_host,ServiceConf.ip_host)
        log.info('pred: %s '% pred)
    
    def valid(self,inputs,labels,n='valid'):
        output_interactice = self.forward(inputs,n,name='valid')
        predictions,valid_loss = self.top_model.valid(output_interactice,labels)
        self.remote(valid_loss,"valid_loss",ServiceConf.port_host,ServiceConf.ip_host)
        log.info('########################################valid_loss: %s#######################################' % valid_loss)
        return valid_loss

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
        XA_train,XB_train,XA_test,XB_test,y_train,y_test = split_data()
        #隐私求交
        # ids = [x for x in range(len(XA_train)-10)]
        # psi_service = PSIService(ids = ids)
        # common_ids = psi_service.run()
        # XB_train ,y_train= XB_train[common_ids],y_train[common_ids]
        #训练
        line_guest = LinrGuest(XB_train,y_train,config)
        early_stopping = EarlyStopping()
        for n in range(config['epoch']):
            line_guest.run(n)
            if n % 50 == 0:
                validate_loss = line_guest.valid(XB_test,y_test)
                early_stopping(validate_loss)
                if early_stopping.early_stop:
                    log.info("Early stopping")
                    line_guest.save()
                    break
                
        #预测
        # line_guest.load_model()
        line_guest.predict(XB_test)
    finally:
        line_guest.fed_lmdb.drop()
    