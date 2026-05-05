import unittest

import response_utils


class ResponseHandlingTests(unittest.TestCase):
    def test_normalizes_blank_model_response_to_fallback_text(self):
        text = response_utils.normalize_response_text("")

        self.assertEqual(text, response_utils.FALLBACK_RESPONSE)

    def test_normalizes_missing_raw_response_text_to_fallback_text(self):
        text = response_utils.extract_raw_response_text({"content": [{}]})

        self.assertEqual(text, response_utils.FALLBACK_RESPONSE)

    def test_tts_sentences_never_include_blank_text(self):
        sentences = response_utils.sentences_for_tts("   ")

        self.assertEqual(sentences, [])

    def test_tts_sentences_split_non_blank_text(self):
        sentences = response_utils.sentences_for_tts("Hello there. How can I help?")

        self.assertEqual(sentences, ["Hello there.", "How can I help?"])

    def test_tts_sentences_merge_short_chinese_punctuation(self):
        sentences = response_utils.sentences_for_tts("你好。我可以帮你吗？可以的！")

        self.assertEqual(sentences, ["你好。我可以帮你吗？可以的！"])

    def test_tts_sentences_merge_short_chinese_segments(self):
        sentences = response_utils.sentences_for_tts("你好！你看起来很开心。有什么我可以帮你的吗？")

        self.assertEqual(sentences, ["你好！你看起来很开心。有什么我可以帮你的吗？"])

    def test_tts_sentences_append_short_final_chinese_segment(self):
        long_sentence = ("这" * (response_utils.MAX_TTS_SEGMENT_CHARS + 5)) + "。"

        sentences = response_utils.sentences_for_tts(f"{long_sentence}好的！")

        self.assertEqual(sentences[-1][-3:], "好的！")
        self.assertGreaterEqual(len(sentences[-1]), response_utils.SHORT_TTS_SEGMENT_CHARS)

    def test_tts_sentences_do_not_split_decimal_numbers(self):
        sentences = response_utils.sentences_for_tts("版本是 3.14。可以继续。")

        self.assertEqual(sentences, ["版本是 3.14。可以继续。"])

    def test_does_not_stream_tts_for_non_tool_fallback(self):
        self.assertFalse(response_utils.should_stream_tts(used_tool=False, text=response_utils.FALLBACK_RESPONSE))

    def test_streams_tts_for_tool_response(self):
        self.assertTrue(response_utils.should_stream_tts(used_tool=True, text="Hello there."))

    def test_detects_litert_tool_parse_errors(self):
        error = RuntimeError("INVALID_ARGUMENT: Failed to parse tool calls from response")

        self.assertTrue(response_utils.is_litert_tool_parse_error(error))

    def test_ignores_unrelated_errors_as_tool_parse_errors(self):
        error = RuntimeError("network write failed")

        self.assertFalse(response_utils.is_litert_tool_parse_error(error))


if __name__ == "__main__":
    unittest.main()
