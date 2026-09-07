"""
Unit tests for Query Normalization and Safe Typo Correction.

Tests:
1. Morphological normalization (singular/plural conversion, unit alias expansion)
2. Safe typo correction (handling spelling errors via trusted vocabulary)
3. Protected tokens (technical codes, model numbers, hyphenated formats, numbers never corrupted)
4. Preservation of original clean_query and raw_query
5. NormalizedQuery structure and fast-path intent handling
"""
import pytest
from app.services.query.normalization import QueryNormalizationService, NormalizedQuery
from app.services.query.typo_correction import (
    build_typo_query,
    _is_protected_token,
    _find_correction,
)


@pytest.fixture
def norm_service():
    return QueryNormalizationService()


class TestMorphologicalNormalization:
    def test_kilograms_to_pounds_normalization(self, norm_service):
        q1 = "What is the conversion factor from kilograms to pounds?"
        res1 = norm_service.normalize(q1)

        # clean_query must preserve user words
        assert "kilograms" in res1.clean_query.lower()
        assert "pounds" in res1.clean_query.lower()

        # retrieval_query must be expanded to include singular forms
        assert "kilogram" in res1.retrieval_query.lower()
        assert "pound" in res1.retrieval_query.lower()
        assert "kilograms" in res1.retrieval_query.lower()
        assert "pounds" in res1.retrieval_query.lower()

    def test_kilogram_singular_equality(self, norm_service):
        q2 = "What is the conversion factor from kilogram to pounds?"
        res2 = norm_service.normalize(q2)
        assert "kilogram" in res2.retrieval_query.lower()
        assert "pound" in res2.retrieval_query.lower()
        assert "pounds" in res2.retrieval_query.lower()

    def test_abbreviation_aliases(self, norm_service):
        q = "Convert 50 kg to lbs and check kpa"
        res = norm_service.normalize(q)
        assert "kilogram" in res.retrieval_query.lower()
        assert "pound" in res.retrieval_query.lower()
        assert "kilopascal" in res.retrieval_query.lower()

    def test_preserves_clean_query_unaltered(self, norm_service):
        q = "  What is the   conversion factor from kilograms to pounds?  "
        res = norm_service.normalize(q)
        assert res.raw_query == q
        assert res.clean_query == "What is the conversion factor from kilograms to pounds?"


class TestSafeTypoCorrection:
    def test_common_unit_typos(self):
        # Common typos of kilogram appended
        assert "kilogram" in build_typo_query("What is the kilogrm factor?")
        assert "kilogram" in build_typo_query("What is the kilogam factor?")

        # Common typos of pound appended
        assert "pound" in build_typo_query("Convert to punds").lower()
        assert "pound" in build_typo_query("Convert to pouds").lower()

    def test_technical_word_typo(self):
        corrected = build_typo_query("Check the machine specifcations")
        assert "specifications" in corrected.lower()
        assert "specifcations" not in corrected.lower()

    def test_no_change_when_correct(self):
        clean = "What is the conversion factor from kilogram to pound?"
        assert build_typo_query(clean) == clean

    def test_capitalization_preserved(self):
        corrected = build_typo_query("Kilogam conversion")
        assert corrected.startswith("Kilogram")


class TestProtectedTokens:
    @pytest.mark.parametrize("token", [
        "R215L",
        "E-104",
        "P2001",
        "2-3",
        "HX220",
        "HX-220",
        "215",
        "2",
        "3.5",
        "RPM",
        "PSI",
        "LBS",
    ])
    def test_protected_tokens_not_fuzzy_matched(self, token):
        assert _is_protected_token(token) is True
        # In query context, protected token is unaltered
        query = f"Check {token} status"
        corrected = build_typo_query(query)
        assert token in corrected

    def test_page_number_notation_preserved(self, norm_service):
        query = 'In the example page number "2-3", what does the "2" represent?'
        res = norm_service.normalize(query)
        assert res.intent == "PAGE_NUMBER_FORMAT"
        assert "2-3" in res.clean_query
        assert "2" in res.clean_query
        # Typos should not corrupt "2-3" or "2"
        assert "2-3" in res.typo_query or res.typo_query == res.clean_query


class TestIntentAndChannelsInNormalizedQuery:
    def test_all_intents_populate_typo_and_retrieval_query(self, norm_service):
        queries = [
            "How many major sections are there in the manual?",  # COUNT_QUERY
            "List the major sections of the manual",            # LIST_QUERY
            "What is section 3?",                               # SECTION_LOOKUP
            "What is the difference between safety and caution symbols?",  # COMPARISON / GENERAL_QA
        ]
        for q in queries:
            res = norm_service.normalize(q)
            assert isinstance(res, NormalizedQuery)
            assert res.raw_query == q
            assert res.clean_query != ""
            assert res.retrieval_query != ""
            assert res.typo_query != ""
