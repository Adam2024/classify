import os

GPU_PATH = "/home/libaokui2021/classify"


def abs_corpus_path(rel_path: str) -> str:
    return os.path.join(GPU_PATH, rel_path)


# data
DATA_ROOT_DIR = abs_corpus_path('')

RAW_CORPUS_DIR = abs_corpus_path('origin_secondary/')

TRAIN_DATA_DIR = abs_corpus_path('data/')

OTHERS_DATA_DIR = abs_corpus_path('others/')

REVISED_DATA_DIR = abs_corpus_path('secondary_corrected_json')

SECOND_BATCH_DIR = abs_corpus_path('商铺租赁二级标签修改版0226')


# model
MODEL_DIR = '/home/li/classify/model'
DEPLOY_MODEL_DIR = '/deployments/model/'
# DEPLOY_MODEL_DIR = MODEL_DIR

PRIAMRY_SINGLE_FILTERED_T = DEPLOY_MODEL_DIR + 'primary_single_filtered'

PRIAMRY_SINGLE_FILTERED = DEPLOY_MODEL_DIR + 'primary_single_filtered_lstm'

PRIMARY_MULTI_FILTERED = DEPLOY_MODEL_DIR + 'primary_with_other_filtered_shuffle'

SECONDARY_MODEL = DEPLOY_MODEL_DIR + 'total_second_batch_lstm'

SECONDARY_MODEL_T = DEPLOY_MODEL_DIR + 'extract_secondary_filtered'


PAYMENT_PART_MODEL = DEPLOY_MODEL_DIR + 'payment_part'

TAX_PAYMENT_PART_MODEL = DEPLOY_MODEL_DIR + 'tax_payment_part_lstm'


# result
TUNE_RESULTS_DIR = '/home/li/classify/result/'


all_need = {
    '首部', '商铺地址', '商铺面积', '商铺用途',
    '租期具体时间', '租金金额', '迟延支付租金违约责任',
    '其他费用承担', '商铺交付时间', '协助办证义务',
    '商铺返还', '疫情情形', '政府行为情形',
}

primary_whitelist = {
    '首部', '商铺信息和用途', '租期', '费用和支付',
    '商铺交付', '商铺返还', '免责',
    '背景说明', '续租', '费用和支付', '保险',
    '商铺交付', '商铺使用', '商铺维修', '商铺维修改造',
    '商铺转让转租', '违约责任', '合同变更和解除', '保密',
    '通知送达', '合同解释', '争议解决', '适用法律',
    '合同组成和份数', '合同生效', '其他', '尾部'
}

primary_using = {
    '首部', '商铺信息和用途', '租期',
    '费用和支付', '商铺交付', '商铺返还',
    '免责'
}

house_info_whitelist = {
    '商铺地址', '商铺面积', '商铺用途',
    '合同所有权人', '有权出租保证'
}

lease_term_whitelist = {
    '租期具体时间', '租期起算日', '迟延交付违约责任'
}

cost_payment_whitelist = {
    '租金和支付', '租金支付时间', '租金支付方式',
    '租金金额', '租金计算标准', '租金调整机制',
    '迟延支付租金违约责任', '税费承担', '税费变更约定',
    '保证金和支付', '保证金金额', '保证金计算标准',
    '保证金支付时间', '迟延支付保证金违约责任',
    '保证金返还时间', '保证金返还条件', '保证金逾期返还违约责任',
    '其他费用和支付', '水费承担', '水费计算标准', '电费承担',
    '电费计算标准', '物业费承担', '其他费用承担',
    '迟延支付其他费用违约责任', '迟延支付兜底违约责任',
    '发票条款', '普通发票', '增值税专用发票', '开票时间',
    '收款账户信息', '正常损害维修'
}

cost_payment_blacklist = {
    '税费变更约定', '保证金逾期返还违约责任',
    '普通发票', '增值税专用发票'
}

delivery_whitelist = {
    '商铺有无抵押', '无抵押保证', '商铺抵押违约责任',
    '未被查封保证', '商铺交付时间', '迟延交房违约责任',
    '协助办证义务'
}

disclaimer_whitelist = {
    '疫情情形', '政府行为情形'
}


sec_blacklist = {
    '迟延交付违约责任', '商铺查封违约责任', '商铺违章建筑违约责任',
    '保证金逾期返还违约责任', '未被查封保证', '无抵押保证',
    '无违章建筑保证', '税费变更约定', '违反一裁终局',
    '仲裁地', '题干', '商铺转让转租', '选项',
    '选择结果-仲裁', '同业租赁', '选择结果-诉讼', '合同生效', '增值税专用发票',
    '商铺抵押违约责任', '待定标签',
    # 一级标签中非互斥项
    '违约责任', '合同变更和解除'
}
