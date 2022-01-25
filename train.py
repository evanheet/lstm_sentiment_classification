import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, num_epochs, learning_rate):
        
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
    def train(self):
        
        opitmizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, batch in enumerate(self.model.train_iter):
                label = batch.label.sub(1)
                opitmizer.zero_grad()
                predictions = self.model(batch.text).squeeze(1)
        
                # Process training loss
                loss = criterion(predictions, label)
                training_loss.append(loss.item())
        
                # Process training accuracy
                correct = (predictions.argmax(dim=1) == label).float()
                accuracy = correct.sum()/len(correct)
                training_accuracy.append(accuracy.item())
        
                loss.backward()
                opitmizer.step()
                if (i+1) % 20 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, self.num_epochs, i+1, self.model.train_iterations, loss.item(), accuracy.item()))
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                epoch_loss = 0
                epoch_accuracy = 0
                for batch in self.model.val_iter:
                    label = batch.label.sub(1)
                    predictions = self.model(batch.text.squeeze(1))
            
                    # Process validation loss
                    loss = criterion(predictions, label)
                    validation_loss.append(loss.item())
                    epoch_loss = epoch_loss + loss
            
                    # Process validation accuracy
                    correct = (predictions.argmax(dim=1) == label).float()
                    accuracy = correct.sum()/len(correct)
                    validation_accuracy.append(accuracy.item())
                    epoch_accuracy = epoch_accuracy + accuracy
            
            print("**********************************************************************************************************")        
            print ('VALIDATION FOR EPOCH {}/{} --> Average Loss: {:.4f}, Average Accuracy: {:.4f}'.format(epoch+1, self.num_epochs, epoch_loss/self.model.val_iterations, epoch_accuracy/self.model.val_iterations))
            print("**********************************************************************************************************")
            
            
    def copyModel(self):
        return self.model