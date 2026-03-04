import logging

logger = logging.getLogger('base')


def create_model(opt):
    from src.pipelines.ddpm_pipeline import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


def create_CD_model(opt):
    from src.pipelines.cd_pipeline import CD as M
    m = M(opt)
    logger.info('CD Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
