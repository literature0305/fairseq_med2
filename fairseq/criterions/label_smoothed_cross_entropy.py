# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import Levenshtein


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

@dataclass
class SegmentLevelTrainingConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@dataclass
class ScheduledSamplingConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@dataclass
class ScheduledSamplingV2Config(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss2(lprobs, target, epsilon, ignore_index=None, reduce=True):
    eps_i = epsilon / (lprobs.size(-1) - 1)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -((1.0 - epsilon - eps_i) * lprobs).gather(dim=-1, index=target)
    smooth_loss = -(eps_i * lprobs).sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    loss = nll_loss + smooth_loss
    return loss, nll_loss


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss2(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion(
    "segment_level_training", dataclass=SegmentLevelTrainingConfig
)
class SegmentLevelTraining(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True, update_num=None, hypos=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # print('1',sample['target'].size())
        # print('2',sample['net_input']['src_tokens'].size())
        # print('3',sample['net_input']['src_lengths'].size())
        # raise ValueError('target', sample['target'])   
        # 1 torch.Size([136, 29])
        # 2 torch.Size([136, 19])
        # 3 torch.Size([136])    
        # ValueError: ('target', tensor([[ 14,  18,  43,  ...,   1,   1,   1],
     

        # get pre tokens - beam search results
        if hypos is not None and self.training:
            # tensor to store beam-search hyps
            use_one_best=True # if False: randomly choice one hypothesis in the hypothesis set
            num_nbest_hyps=3 # TODO: remove hard coded tuning factor1

            # scheduling 2
            update_num_max = 500000 # (BLEU ?)
            ter_threshold = 0.3 + 0.5*min(1, update_num / update_num_max)

            remove_set = {1,2} # remove padding, EOS
            prev_output_tokens_new = torch.ones(sample['target'].size(0), num_nbest_hyps, sample['target'].size(1) * 2).to(sample['target'].device).to(sample['target'].dtype)
            prev_output_tokens_new[:,:,0] = prev_output_tokens_new[:,:,0] * 2 # SOS:2, PAD:1 # TODO: remove hard coded tuning factor3
            target_new = sample['target'].clone()
            src_tokens_new = sample['net_input']['src_tokens'].clone()
            src_lengths_new = sample['net_input']['src_lengths'].clone()
            
            max_len_hyp = 0
            num_accepted=0

            for idx in range(len(hypos)):
                list_true = sample['target'][idx].tolist()
                if max_len_hyp < len(list_true) + 1:
                    max_len_hyp = len(list_true) + 1
                list_true = [i for i in list_true if i not in remove_set]

                if use_one_best:
                    n_th_best = 0
                else:
                    n_th_best = torch.randperm(len(hypos[idx]))[0]
                accepted_hyps = []
                for idx_hyp_tmp in range(len(hypos[idx])):
                    idx_hyp = (idx_hyp_tmp+n_th_best) % len(hypos[idx])
                    list_hyp = hypos[idx][idx_hyp]['tokens'].tolist()
                    list_hyp = [i for i in list_hyp if i not in remove_set]
                    if max_len_hyp < len(list_hyp) + 1:
                        max_len_hyp = len(list_hyp) + 1
                    distance = Levenshtein.distance(list_true, list_hyp)
                    token_error_rate = distance / len(list_true) ###################################
                    if token_error_rate < ter_threshold:
                        accepted_hyps.append(list_hyp)

                if len(accepted_hyps) == 0:
                    pass
                else:
                    for idx_hyp_tmp in range(len(hypos[idx])):
                        idx_hyp = idx_hyp_tmp % len(accepted_hyps)

                        ## TODO: you can remove this option
                        if len(accepted_hyps[idx_hyp]) < sample['target'].size(1) * 2:
                            prev_output_tokens_new[num_accepted][idx_hyp_tmp][1:len(accepted_hyps[idx_hyp])+1] = torch.tensor(accepted_hyps[idx_hyp]).to(prev_output_tokens_new.device)
                        else:
                            end_point = sample['target'].size(1) * 2 - 1
                            prev_output_tokens_new[num_accepted][idx_hyp_tmp][1:] = torch.tensor(accepted_hyps[idx_hyp][:end_point]).to(prev_output_tokens_new.device)

                    target_new[num_accepted] = sample['target'][idx]
                    src_tokens_new[num_accepted]=sample['net_input']['src_tokens'][idx]
                    src_lengths_new[num_accepted]=sample['net_input']['src_lengths'][idx]
                    
                    num_accepted = num_accepted + 1

            prev_output_tokens_new = prev_output_tokens_new[:,:,:max_len_hyp]
            if use_one_best:
                random_idx = 0
            else:
                random_idx = torch.randperm(num_nbest_hyps)[0]

            # get random_idx th hyp in hypothesis set
            prev_output_tokens_new = prev_output_tokens_new[:,random_idx]

            # remove redundancy
            if num_accepted > 0:
                prev_output_tokens_new = prev_output_tokens_new[:num_accepted]
                target_new = target_new[:num_accepted]
                src_lengths_new = src_lengths_new[:num_accepted]
                src_tokens_new = src_tokens_new[:num_accepted]
        else:
            # prev_output_tokens_new = sample['net_input']['prev_output_tokens']
            pass
        
        if hypos is not None and self.training:
            if num_accepted != 0:
                # forward with beam-search hyps
                net_output = model(src_tokens_new, src_lengths_new, prev_output_tokens_new)

                # get segment-level loss with random BLEU size
                bleu_size_random = torch.randperm(4)[0]
                loss, nll_loss = self.compute_loss_bleu4(model, net_output, target_new, prev_output_tokens_new, reduce=reduce, bleu_size=bleu_size_random+2)
            else:
                loss=0
                nll_loss=0

            # get original loss
            net_output = model(**sample["net_input"])
            loss2, nll_loss2 = self.compute_loss(model, net_output, sample, reduce=reduce)

            if torch.randperm(500)[0]==0:
                print('ter_threshold:', ter_threshold)
                print('num_accepted / total', num_accepted, len(sample['target']))
                print('loss token level:', loss2)
                print('loss seque level:', loss)
            
            # segment-level loss + maximum likelihood loss
            loss = loss + loss2
            nll_loss = nll_loss + nll_loss2
        else:
            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if torch.randperm(200)[0]==0:
            print('lprobs.size():', lprobs.size())

        loss, nll_loss = label_smoothed_nll_loss2(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def align_bleu4(self, ys_out_pad, target_original, conf_argmax, len_max_time, threshold_distance, idx_batch, bleu_size):
        ################ input ################
        # ys_out_pad: [B,U], target batch
        # target_original: [u_b], target label sequence
        # conf_argmax: [l_b], beam-search decoding results
        # len_max_time: max time to generate batched target tensor
        # if edit distance < threshold_distance: use it for train
        # bleu_size: segment size
        ################ output ################
        # target_after_utt:
        # idx_conf_after_utt:
        # len_max_time:

        target_after_utt = []
        idx_conf_after_utt = []
        len_time = 0
        for idx_tokens in range(len(target_original)-bleu_size):
            sub_seq = target_original[idx_tokens:idx_tokens+bleu_size]
            shotest_distance=999
            best_sub_seq_to_append_target=[]
            best_sub_seq_to_append_idx_conf=[]
            for idx_tokens2 in range(len(conf_argmax)-bleu_size):
                sub_seq2 = conf_argmax[idx_tokens2:idx_tokens2+bleu_size]
                distance = Levenshtein.distance(sub_seq, sub_seq2)
                if (distance < threshold_distance + 1) and (distance < shotest_distance + 1):
                    shotest_distance = distance
                    edit_log=Levenshtein.editops(sub_seq,sub_seq2)
                    idx_conf = list(range(idx_tokens2, idx_tokens2 + bleu_size))
                    target_after = list(sub_seq)
                    idx_conf_after=list(idx_conf)
                    for ele in edit_log:
                        err=ele[0]
                        position_a = ele[1]
                        position_b = ele[2]
                        if err == 'insert':
                            if position_a < len(sub_seq):
                                target_after.insert(position_a, sub_seq[position_a])
                            else:
                                target_after.insert(position_a, sub_seq[-1])
                        elif err == 'delete':
                            if position_b < len(idx_conf):
                                idx_conf_after.insert(position_b, idx_conf[position_b])
                            else:
                                idx_conf_after.insert(position_b, idx_conf[-1])
                    assert len(target_after) == len(idx_conf_after)

                    best_sub_seq_to_append_target = target_after
                    best_sub_seq_to_append_idx_conf = idx_conf_after
            # append target_after that have shortest distance(sub_seq2, sub_seq)                
            len_time = len_time + len(best_sub_seq_to_append_idx_conf)
            idx_conf_after_utt.extend(best_sub_seq_to_append_idx_conf) # .extend(pred_pad[idx_batch][idx_conf_after])
            target_after_utt.extend(torch.tensor(best_sub_seq_to_append_target).to(ys_out_pad.device).to(ys_out_pad.dtype))

        if len_time == 0:
            if bleu_size > 1:
                return self.align_bleu4(ys_out_pad, target_original, conf_argmax, len_max_time, threshold_distance, idx_batch, bleu_size-1)
            else:
                return torch.zeros(1).to(ys_out_pad.device).to(ys_out_pad.dtype), pred_pad[idx_batch][0]
        elif len_time > len_max_time:
            len_max_time = len_time

        return target_after_utt, idx_conf_after_utt, len_max_time

    def compute_loss_bleu4(self, model, net_output, ys_out_pad, prev_output_tokens_new, reduce=True, bleu_size=4):
        pred_pad = torch.log_softmax(net_output[0], dim=-1)
        # ys_out_pad = sample['target']
        conf_argmax_batch=torch.argmax(pred_pad.detach(),dim=-1)
        len_ys_in_pad = (prev_output_tokens_new != self.padding_idx).sum(-1) + 1 # B,V -> B

        len_target = (ys_out_pad != self.padding_idx).sum(-1)
        pred_pad_after_batch = []
        target_after_batch = []
        len_max_time = 1
        threshold_distance = bleu_size // 2 # 2 # TODO: remove hard coded tuning factor4

        for idx_batch, ys_out in enumerate(ys_out_pad):
            target_original = ys_out.tolist()[:len_target[idx_batch]] # len(non_pad_labels)
            conf_argmax = conf_argmax_batch[idx_batch].tolist()[:len_ys_in_pad[idx_batch]]

            # get alignment between output log-probability & target
            target_after_utt, idx_conf_after_utt, len_max_time = self.align_bleu4(ys_out_pad, target_original, conf_argmax, len_max_time, threshold_distance, idx_batch, bleu_size)
            target_after_batch.append(torch.tensor(target_after_utt).to(ys_out_pad.device).to(ys_out_pad.dtype))
            pred_pad_after_batch.append(pred_pad[idx_batch][idx_conf_after_utt])

        # pad
        ys_out_pad_new = torch.zeros(len(ys_out_pad), len_max_time).to(ys_out_pad.device).to(ys_out_pad.dtype) # * (self.padding_idx)
        pred_pad_new = torch.zeros(pred_pad.size(0), len_max_time, pred_pad.size(-1)).to(pred_pad.device).to(pred_pad.dtype)
        for idx_pred_pad in range(len(pred_pad_new)):
            len_pred = len(pred_pad_after_batch[idx_pred_pad])
            len_target = len(target_after_batch[idx_pred_pad])
            pred_pad_new[idx_pred_pad][:len_pred] = pred_pad_new[idx_pred_pad][:len_pred] + pred_pad_after_batch[idx_pred_pad]
            ys_out_pad_new[idx_pred_pad][:len_target] = ys_out_pad_new[idx_pred_pad][:len_target] + target_after_batch[idx_pred_pad]
            ys_out_pad_new[idx_pred_pad][len_target:] = ys_out_pad_new[idx_pred_pad][len_target:] + self.padding_idx

        pred_pad_new = pred_pad_new.view(-1, pred_pad_new.size(-1))
        ys_out_pad_new = ys_out_pad_new.view(-1)

        if torch.randperm(200)[0]==0:
            print('pred_new.size():', pred_pad_new.size())

        # calculate loss
        loss, nll_loss = label_smoothed_nll_loss2(
            pred_pad_new,
            ys_out_pad_new,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

@register_criterion(
    "scheduled_sampling", dataclass=ScheduledSamplingConfig
)
class ScheduledSamplingCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True, update_num=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        stochastic_scheduling = False
        # TODO: remove hard coded tuning factor5

        # option 1.0 without-scheduling
        # if torch.randperm(10)[0] < 1:
        #     stochastic_scheduling = True

        # option 2.0 with-scheduling
        update_num_max = 50000
        sampling_prob_max = 0.1
        if update_num is not None:
            sampling_prob = min(sampling_prob_max, sampling_prob_max*update_num / update_num_max)
        else:
            sampling_prob = 0
        if torch.randperm(100)[0] / 100 < sampling_prob: 
            stochastic_scheduling = True

        if update_num is not None and update_num > 5000 and stochastic_scheduling:
            hyps = torch.argmax(net_output[0], dim=-1)
            prev_output_tokens_new = sample['net_input']['prev_output_tokens'].clone()
            prev_output_tokens_new[:,1:] = hyps[:,:-1]
            mask = (sample['net_input']['prev_output_tokens'] == 1).to(sample['net_input']['prev_output_tokens'].device).to(sample['net_input']['prev_output_tokens'].dtype)
            prev_output_tokens_new = prev_output_tokens_new * (1-mask) + mask
            net_output = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], prev_output_tokens_new)
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            if torch.randperm(200)[0] == 0:
                total_num = torch.ones(prev_output_tokens_new.size()).sum()
                correct = (prev_output_tokens_new == sample['net_input']['prev_output_tokens']).sum()
                ratio  = correct / total_num
                print('correct ratio:', ratio)
                print('HYP:', prev_output_tokens_new[0])
                print('TAR:', sample['net_input']['prev_output_tokens'][0])
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss2(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True



@register_criterion(
    "scheduled_sampling_v2", dataclass=ScheduledSamplingV2Config
)
class ScheduledSamplingCriterionV2(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True, update_num=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        # TODO: remove hard coded tuning factor5

        # option 1.0 without-scheduling
        # if torch.randperm(10)[0] < 4:

        # option 2.0 with-scheduling
        update_num_max = 150000
        sampling_prob_max = 0.3
        if update_num is not None:
            sampling_prob = min(sampling_prob_max, sampling_prob_max*update_num / update_num_max)
        else:
            sampling_prob = 0

        if update_num is not None and update_num > 5000:
            hyps = torch.argmax(net_output[0], dim=-1)
            prev_output_tokens_new = sample['net_input']['prev_output_tokens'].clone()
            prev_output_tokens_new[:,1:] = hyps[:,:-1]

            mask_ss = torch.rand(prev_output_tokens_new.size())
            mask_ss = (mask_ss < sampling_prob).to(prev_output_tokens_new.dtype).to(prev_output_tokens_new.device)
            prev_output_tokens_new = prev_output_tokens_new * (mask_ss) + sample['net_input']['prev_output_tokens'] * (1-mask_ss)

            mask = (sample['net_input']['prev_output_tokens'] == 1).to(sample['net_input']['prev_output_tokens'].device).to(sample['net_input']['prev_output_tokens'].dtype)
            prev_output_tokens_new = prev_output_tokens_new * (1-mask) + mask
            net_output = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], prev_output_tokens_new)
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            if torch.randperm(200)[0] == 0:
                total_num = torch.ones(prev_output_tokens_new.size()).sum()
                correct = (prev_output_tokens_new == sample['net_input']['prev_output_tokens']).sum()
                ratio  = correct / total_num
                print('correct ratio:', ratio)
                print('HYP:', prev_output_tokens_new[0])
                print('TAR:', sample['net_input']['prev_output_tokens'][0])
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss2(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True



