from framework.evaluators.frame_evaluator import FrameEvaluator


def init_evaluator(name, model, flip):
    return FrameEvaluator(model,flip)

