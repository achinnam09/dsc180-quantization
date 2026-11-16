import pytest

from evaluation.longbench_metrics import (
    classification_score,
    code_sim_score,
    normalize_answer,
    qa_f1_score,
    rouge_score,
    score_dataset,
    score_sample,
)


def test_normalize_answer_drops_case_punctuation_articles():
    assert normalize_answer("The South-West Ultras, a club!") == "southwest ultras club"


def test_qa_f1_by_hand():
    # pred tokens: [most, active, club, is, south, west, ultras] (7)
    # gold tokens: [south, west, ultras, fan, club] (5); overlap = south, west, ultras, club = 4
    # precision 4/7, recall 4/5, F1 = 2PR/(P+R) = 2/3
    score = qa_f1_score("The most active club is the South West Ultras.", "South West Ultras fan club.")
    assert score == pytest.approx(2 / 3)


def test_qa_f1_no_overlap_is_zero():
    assert qa_f1_score("unanswerable", "BERT") == 0


def test_score_sample_takes_best_reference():
    assert score_sample("qasper", "BERT", ["something else", "BERT"], None) == 1.0


def test_first_line_only_for_few_shot_tasks():
    pred = "Paris\nQuestion: what is ...\nAnswer: London"
    assert score_sample("triviaqa", pred, ["Paris"], None) == 1.0  # later lines ignored
    assert score_sample("qasper", pred, ["Paris"], None) < 1.0  # whole prediction scored


def test_classification_score():
    classes = ["Abbreviation", "Animal", "Description", "City"]
    assert classification_score("Animal", "Animal", all_classes=classes) == 1.0
    assert classification_score("City or Animal", "Animal", all_classes=classes) == 0.5
    assert classification_score("City", "Animal", all_classes=classes) == 0.0


def test_code_sim_uses_first_non_comment_line():
    pred = "# comment\nreturn x + 1\nmore"
    assert code_sim_score(pred, "return x + 1") == 1.0


def test_rouge_identical_and_empty():
    assert rouge_score("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    assert rouge_score("", "the cat") == 0.0  # official code returns 0 on rouge errors


def test_score_dataset_is_mean_times_100():
    assert score_dataset("qasper", ["BERT", "nope"], [["BERT"], ["BERT"]], None) == 50.0
