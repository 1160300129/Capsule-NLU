import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, Margin

margin = Margin()


def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name and 'batch' not in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0
    dev_best_intent_loss = float('inf')
    dev_best_slot_loss = float('inf')
    last_improve = 0
    flag = False
    for epoch in range(config.max_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.max_epochs))
        for i, (trains, labels, slot) in enumerate(train_iter):
            if trains[0].shape[0] == 0 or labels.shape[0] == 0:
                continue
            outputs = model(trains)
            slot_outputs = outputs[0]
            intent_outputs = outputs[1]
            # slot_routing_weight = outputs[2]
            # intent_routing_weight = outputs[3]
            model.zero_grad()
            slot = slot.view(-1)
            intent_1 = torch.max(intent_outputs, dim=-1, keepdim=False)[0].cuda()
            # intent_onehot = torch.zeros(config.batch_size,config.intent_size).cuda()
            # loss_intent = margin(intent_outputs, intent_onehot).cuda()
            # loss_intent = torch.mean(loss_intent, dim=-1).sum(dim=-1)
            loss_intent = F.cross_entropy(intent_1, labels)
            loss_slot = F.cross_entropy(slot_outputs, slot)
            loss = loss_intent + loss_slot
            loss.backward()

            optimizer.step()
            scheduler.step(epoch=epoch)

            if total_batch % 100 == 0:
                true_intent = labels.data.cpu()
                true_slot = slot.data.cpu()
                predict_intent = torch.max(intent_1.data, 1)[1].cpu()
                predict_slot = torch.max(slot_outputs.data, 1)[1].cpu()
                train_acc_intent = metrics.accuracy_score(true_intent, predict_intent)
                train_acc_slot = metrics.accuracy_score(true_slot, predict_slot)
                dev_intent_acc, dev_intent_loss, dev_slot_acc, dev_slot_loss = evaluate(config, model, dev_iter)
                if dev_intent_loss < dev_best_intent_loss or dev_slot_loss < dev_best_slot_loss:
                    dev_best_intent_loss = dev_intent_loss
                    dev_best_slot_loss = dev_slot_loss
                    torch.save(model.state_dict(), 'model/' + config.run_name + '.ckpt')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg_intent = 'Iter: {0:>6},  Train Intent Loss: {1:>5.2},  Train Intent Acc: {2:>6.2%},  ' \
                             'Val Intent Loss: {3:>5.2},  Val Intent Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg_intent.format(total_batch, loss_intent.item(), train_acc_intent, dev_intent_loss,
                                        dev_intent_acc, time_dif, improve))
                msg_slot = 'Iter: {0:>6},  Train Slot Loss: {1:>5.2},  Train Slot Acc: {2:>6.2%},  ' \
                           'Val Slot Loss: {3:>5.2},  Val Slot Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg_slot.format(total_batch, loss_slot.item(), train_acc_slot, dev_slot_loss, dev_slot_acc,
                                      time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > 1500:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load('model/' + config.run_name + '.ckpt'))
    model.eval()
    start_time = time.time()
    test_acc_intent, test_intent_loss, test_intent_report, test_intent_confusion, test_slot_loss, test_acc_slot, test_slot_confusion = evaluate(
        config, model, test_iter, test=True)
    msg_intent = 'Test Intent Loss: {0:>5.2},  Test Intent Acc: {1:>6.2%}'
    msg_slot = 'Test Slot Loss: {0:>5.2},  Test Slot Acc: {1:>6.2%}'
    print(msg_intent.format(test_intent_loss, test_acc_intent))
    print(msg_slot.format(test_slot_loss, test_acc_slot))
    print("Precision, Recall and F1-score...")
    print(test_intent_report)
    print("Confusion Matrix...")
    print(test_intent_confusion)
    print(test_slot_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage: ", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_intent_total = 0
    loss_slot_total = 0
    predict_slot_all = np.array([], dtype=int)
    predict_intent_all = np.array([], dtype=int)
    labels_slot_all = np.array([], dtype=int)
    labels_intent_all = np.array([], dtype=int)
    with torch.no_grad():
        i = 0
        for texts, labels, slot in data_iter:
            # print(i)
            if texts[0].shape[0] == 0 or labels.shape[0] == 0:
                continue
            outputs = model(texts)
            slot_outputs = outputs[0]
            intent_outputs = outputs[1]
            slot = slot.view(-1)
            # loss_intent = F.multi_margin_loss(intent_outputs, labels)
            intent_1 = torch.max(intent_outputs, dim=-1, keepdim=False)[0].cuda()
            loss_intent = F.cross_entropy(intent_1, labels)
            loss_slot = F.cross_entropy(slot_outputs, slot)
            loss_slot_total += loss_slot
            loss_intent_total += loss_intent
            labels = labels.data.cpu().numpy()
            slot = slot.data.cpu().numpy()
            predict_intent = torch.max(intent_1.data, 1)[1].cpu()
            predict_slot = torch.max(slot_outputs.data, 1)[1].cpu()
            labels_intent_all = np.append(labels_intent_all, labels)
            labels_slot_all = np.append(labels_slot_all, slot)
            predict_intent_all = np.append(predict_intent_all, predict_intent)
            predict_slot_all = np.append(predict_slot_all, predict_slot)
            i += 1
    acc_intent = metrics.accuracy_score(labels_intent_all, predict_intent_all)
    new_labels_slot_all = []
    new_predict_slot_all = []
    for a, b in zip(labels_slot_all, predict_slot_all):
        if a == b and a == 72:
            continue
        else:
            new_labels_slot_all.append(a)
            new_predict_slot_all.append(b)
    new_labels_slot_all = np.array(new_labels_slot_all)
    new_predict_slot_all = np.array(new_predict_slot_all)
    acc_slot = metrics.accuracy_score(new_labels_slot_all, new_predict_slot_all)
    if test:
        import os
        from utils import load_vocabulary
        # slot_vocab = load_vocabulary(os.path.join(config.vocab_path, 'test_slot_vocab'))
        # slot_vocab['rev'] = slot_vocab['rev'][0:72]
        intent_vocab = load_vocabulary(os.path.join(config.vocab_path, 'intent_vocab'))
        report_intent = metrics.classification_report(labels_intent_all, predict_intent_all,
                                                      target_names=intent_vocab['rev'], digits=4)
        # report_slot = metrics.classification_report(new_labels_slot_all, new_predict_slot_all,
        #                                             target_names=slot_vocab['rev'], digits=4)
        # print(report_slot)
        confusion_intent = metrics.confusion_matrix(labels_intent_all, predict_intent_all)
        confusion_slot = metrics.confusion_matrix(new_labels_slot_all, new_predict_slot_all)
        return acc_intent, loss_intent_total / len(data_iter), report_intent, confusion_intent, loss_slot_total / len(
            data_iter), acc_slot,  confusion_slot
    return acc_intent, loss_intent_total / len(data_iter), acc_slot, loss_slot_total / len(data_iter)
