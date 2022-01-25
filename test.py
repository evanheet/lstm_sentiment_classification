import torch
import torch.nn as nn

class Tester:
    def __init__(self, model):
        
        # Testing
        criterion = nn.CrossEntropyLoss()
        testing_loss = 0
        testing_accuracy = 0
        average_testing_loss = 0
        average_testing_accuracy = 0
        model.eval()
        with torch.no_grad():
            for batch in model.test_iter:
                label = batch.label.sub(1)
                predictions = model(batch.text.squeeze(1))
        
                # Process testing loss
                loss = criterion(predictions, label)
                testing_loss = testing_loss + loss.item()
        
                # Process testing accuracy
                correct = (predictions.argmax(dim=1) == label).float()
                accuracy = correct.sum()/len(correct)
                testing_accuracy = testing_accuracy + accuracy.item()
        
            average_testing_loss = testing_loss/len(model.test_iter)
            average_testing_accuracy  = (testing_accuracy/len(model.test_iter))*100
    
        print('Average testing loss: {:.4f}, Average testing accuracy: {:.2f}%'.format(average_testing_loss, average_testing_accuracy))