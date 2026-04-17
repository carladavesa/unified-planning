(define (domain compiledproblem-domain)
 (:requirements :strips :negative-preconditions :disjunctive-preconditions :conditional-effects)
 (:predicates 
             (c_0_-1)
             (c_0_0)
             (c_0_1)
             (c_0_2)
             (c_0_3)
             (c_0_4)
             (z_0_0)
             (z_0_1)
             (z_0_2)
             (z_0_3)
             (z_0_4)
             (c_1_-1)
             (c_1_0)
             (c_1_1)
             (c_1_2)
             (c_1_3)
             (c_1_4)
             (z_1_0)
             (z_1_1)
             (z_1_2)
             (z_1_3)
             (z_1_4)
             (c_2_-1)
             (c_2_0)
             (c_2_1)
             (c_2_2)
             (c_2_3)
             (c_2_4)
             (z_2_0)
             (z_2_1)
             (z_2_2)
             (z_2_3)
             (z_2_4)
             (c_3_-1)
             (c_3_0)
             (c_3_1)
             (c_3_2)
             (c_3_3)
             (c_3_4)
             (z_3_0)
             (z_3_1)
             (z_3_2)
             (z_3_3)
             (z_3_4)
             (c_4_-1)
             (c_4_0)
             (c_4_1)
             (c_4_2)
             (c_4_3)
             (c_4_4)
             (z_4_0)
             (z_4_1)
             (z_4_2)
             (z_4_3)
             (z_4_4)
             (c_5_-1)
             (c_5_0)
             (c_5_1)
             (c_5_2)
             (c_5_3)
             (c_5_4)
             (z_5_0)
             (z_5_1)
             (z_5_2)
             (z_5_3)
             (z_5_4)
             (c_6_-1)
             (c_6_0)
             (c_6_1)
             (c_6_2)
             (c_6_3)
             (c_6_4)
             (z_6_0)
             (z_6_1)
             (z_6_2)
             (z_6_3)
             (z_6_4)
             (c_7_-1)
             (c_7_0)
             (c_7_1)
             (c_7_2)
             (c_7_3)
             (c_7_4)
             (z_7_0)
             (z_7_1)
             (z_7_2)
             (z_7_3)
             (z_7_4)
             (c_8_-1)
             (c_8_0)
             (c_8_1)
             (c_8_2)
             (c_8_3)
             (c_8_4)
             (z_8_0)
             (z_8_1)
             (z_8_2)
             (z_8_3)
             (z_8_4)
             (c_9_-1)
             (c_9_0)
             (c_9_1)
             (c_9_2)
             (c_9_3)
             (c_9_4)
             (z_9_0)
             (z_9_1)
             (z_9_2)
             (z_9_3)
             (z_9_4)
             (c_10_-1)
             (c_10_0)
             (c_10_1)
             (c_10_2)
             (c_10_3)
             (c_10_4)
             (z_10_0)
             (z_10_1)
             (z_10_2)
             (z_10_3)
             (z_10_4)
             (c_11_-1)
             (c_11_0)
             (c_11_1)
             (c_11_2)
             (c_11_3)
             (c_11_4)
             (z_11_0)
             (z_11_1)
             (z_11_2)
             (z_11_3)
             (z_11_4)
             (c_12_-1)
             (c_12_0)
             (c_12_1)
             (c_12_2)
             (c_12_3)
             (c_12_4)
             (z_12_0)
             (z_12_1)
             (z_12_2)
             (z_12_3)
             (z_12_4)
             (c_13_-1)
             (c_13_0)
             (c_13_1)
             (c_13_2)
             (c_13_3)
             (c_13_4)
             (z_13_0)
             (z_13_1)
             (z_13_2)
             (z_13_3)
             (z_13_4)
             (c_14_-1)
             (c_14_0)
             (c_14_1)
             (c_14_2)
             (c_14_3)
             (c_14_4)
             (z_14_0)
             (z_14_1)
             (z_14_2)
             (z_14_3)
             (z_14_4)
             (c_15_-1)
             (c_15_0)
             (c_15_1)
             (c_15_2)
             (c_15_3)
             (c_15_4)
             (z_15_0)
             (z_15_1)
             (z_15_2)
             (z_15_3)
             (z_15_4)
             (c_16_-1)
             (c_16_0)
             (c_16_1)
             (c_16_2)
             (c_16_3)
             (c_16_4)
             (z_16_0)
             (z_16_1)
             (z_16_2)
             (z_16_3)
             (z_16_4)
             (c_17_-1)
             (c_17_0)
             (c_17_1)
             (c_17_2)
             (c_17_3)
             (c_17_4)
             (z_17_0)
             (z_17_1)
             (z_17_2)
             (z_17_3)
             (z_17_4)
             (c_18_-1)
             (c_18_0)
             (c_18_1)
             (c_18_2)
             (c_18_3)
             (c_18_4)
             (z_18_0)
             (z_18_1)
             (z_18_2)
             (z_18_3)
             (z_18_4)
             (c_19_-1)
             (c_19_0)
             (c_19_1)
             (c_19_2)
             (c_19_3)
             (c_19_4)
             (z_19_0)
             (z_19_1)
             (z_19_2)
             (z_19_3)
             (z_19_4)
             (c_20_-1)
             (c_20_0)
             (c_20_1)
             (c_20_2)
             (c_20_3)
             (c_20_4)
             (z_20_0)
             (z_20_1)
             (z_20_2)
             (z_20_3)
             (z_20_4)
             (c_21_-1)
             (c_21_0)
             (c_21_1)
             (c_21_2)
             (c_21_3)
             (c_21_4)
             (z_21_0)
             (z_21_1)
             (z_21_2)
             (z_21_3)
             (z_21_4)
             (false)
             (harmony_entertainment_0)
             (harmony_entertainment_1)
             (harmony_entertainment_2)
             (harmony_entertainment_3)
             (harmony_entertainment_4)
             (locale_onion_0)
             (locale_onion_1)
             (locale_onion_2)
             (locale_onion_3)
             (locale_onion_4)
             (v0_0)
             (v0_1)
             (v0_2)
             (v0_3)
             (v0_4)
             (locale_cherry_0)
             (locale_cherry_1)
             (locale_cherry_2)
             (locale_cherry_3)
             (locale_cherry_4)
             (v1_0)
             (v1_1)
             (v1_2)
             (v1_3)
             (v1_4)
             (v2_0)
             (v2_1)
             (v2_2)
             (v2_3)
             (v2_4)
             (harmony_satiety_0)
             (harmony_satiety_1)
             (harmony_satiety_2)
             (harmony_satiety_3)
             (harmony_satiety_4)
             (v3_0)
             (v3_1)
             (v3_2)
             (v3_3)
             (v3_4)
             (harmony_stimulation_0)
             (harmony_stimulation_1)
             (harmony_stimulation_2)
             (harmony_stimulation_3)
             (harmony_stimulation_4)
             (v4_0)
             (v4_1)
             (v4_2)
             (v4_3)
             (v4_4)
             (v5_0)
             (v5_1)
             (v5_2)
             (v5_3)
             (v5_4)
             (harmony_curiosity_0)
             (harmony_curiosity_1)
             (harmony_curiosity_2)
             (harmony_curiosity_3)
             (harmony_curiosity_4)
             (v6_0)
             (v6_1)
             (v6_2)
             (v6_3)
             (v6_4)
             (v7_0)
             (v7_1)
             (v7_2)
             (v7_3)
             (v7_4)
             (locale_beef_0)
             (locale_beef_1)
             (locale_beef_2)
             (locale_beef_3)
             (locale_beef_4)
             (v8_0)
             (v8_1)
             (v8_2)
             (v8_3)
             (v8_4)
             (locale_muffin_0)
             (locale_muffin_1)
             (locale_muffin_2)
             (locale_muffin_3)
             (locale_muffin_4)
             (locale_ham_0)
             (locale_ham_1)
             (locale_ham_2)
             (locale_ham_3)
             (locale_ham_4)
             (locale_flounder_0)
             (locale_flounder_1)
             (locale_flounder_2)
             (locale_flounder_3)
             (locale_flounder_4)
             (locale_tuna_0)
             (locale_tuna_1)
             (locale_tuna_2)
             (locale_tuna_3)
             (locale_tuna_4)
             (v9_0)
             (v9_1)
             (v9_2)
             (v9_3)
             (v9_4)
             (v10_0)
             (v10_1)
             (v10_2)
             (v10_3)
             (v10_4)
             (craves_loneliness_muffin)
             (craves_loneliness_ham)
             (fears_dread_entertainment)
             (fears_loneliness_stimulation)
             (fears_loneliness_curiosity)
             (fears_abrasion_satiety)
             (fears_loneliness_entertainment)
             (fears_abrasion_stimulation)
             (craves_laceration_tuna)
             (craves_dread_onion)
             (craves_prostatitis_beef)
             (craves_boils4_ham)
             (craves_entertainment_ham)
             (craves_satiety_onion)
             (craves_stimulation_flounder)
             (craves_curiosity_cherry)
             (craves_jealousy_flounder)
             (craves_loneliness_onion)
             (craves_anxiety_flounder)
             (craves_sciatica_tuna)
             (craves_grief_beef)
             (craves_grief7_ham)
             (craves_jealousy2_cherry)
             (craves_angina_onion)
             (craves_depression1_muffin)
             (craves_anger_beef)
             (craves_abrasion_beef)
             (craves_hangover_tuna)
             (craves_boils_tuna)
             (craves_prostatitis3_ham)
             (craves_dread8_ham)
             (craves_depression_beef)
             (craves_grief7_beef)
             (craves_depression1_beef)
             (craves_satiety_muffin)
             (craves_stimulation_onion)
             (craves_stimulation_muffin)
             (craves_curiosity_onion)
             (craves_curiosity_muffin)
             (craves_entertainment_onion)
             (craves_entertainment_muffin)
             (craves_satiety_tuna)
             (craves_stimulation_tuna)
             (craves_curiosity_tuna)
             (craves_entertainment_tuna)
             (craves_satiety_ham)
             (craves_stimulation_ham)
             (craves_curiosity_ham)
             (craves_satiety_flounder)
             (craves_curiosity_flounder)
             (craves_entertainment_flounder)
             (craves_satiety_cherry)
             (craves_stimulation_cherry)
             (craves_entertainment_cherry)
             (craves_satiety_beef)
             (craves_stimulation_beef)
             (craves_curiosity_beef)
             (craves_entertainment_beef)
             (fears_abrasion_curiosity)
             (fears_grief_satiety)
             (craves_abrasion_onion)
             (craves_grief_onion)
             (craves_abrasion_tuna)
             (craves_grief_tuna)
             (craves_abrasion_flounder)
             (craves_grief_flounder)
             (craves_abrasion_cherry)
             (craves_grief_cherry)
             (craves_abrasion_muffin)
             (craves_grief_muffin)
             (craves_abrasion_ham)
             (craves_grief_ham)
             (fears_boils4_satiety)
             (craves_boils4_onion)
             (craves_boils4_beef)
             (craves_boils4_flounder)
             (craves_boils4_tuna)
             (craves_boils4_muffin)
             (craves_boils4_cherry)
             (fears_abrasion_entertainment)
             (fears_grief_stimulation)
             (fears_boils4_stimulation)
             (fears_grief_curiosity)
             (fears_prostatitis_satiety)
             (craves_prostatitis_onion)
             (craves_prostatitis_tuna)
             (craves_prostatitis_flounder)
             (craves_prostatitis_cherry)
             (craves_prostatitis_muffin)
             (craves_prostatitis_ham)
             (fears_prostatitis3_satiety)
             (craves_prostatitis3_beef)
             (fears_boils4_curiosity)
             (craves_prostatitis3_tuna)
             (craves_prostatitis3_onion)
             (craves_prostatitis3_cherry)
             (craves_prostatitis3_flounder)
             (craves_prostatitis3_muffin)
             (fears_grief_entertainment)
             (fears_prostatitis_stimulation)
             (fears_boils4_entertainment)
             (fears_prostatitis3_stimulation)
             (fears_prostatitis_curiosity)
             (fears_depression_satiety)
             (craves_depression_onion)
             (craves_depression_tuna)
             (craves_depression_flounder)
             (craves_depression_cherry)
             (craves_depression_muffin)
             (craves_depression_ham)
             (fears_prostatitis3_curiosity)
             (fears_dread8_satiety)
             (craves_dread8_onion)
             (craves_dread8_beef)
             (craves_dread8_flounder)
             (craves_dread8_tuna)
             (craves_dread8_muffin)
             (craves_dread8_cherry)
             (fears_prostatitis_entertainment)
             (fears_depression_stimulation)
             (fears_dread8_stimulation)
             (fears_prostatitis3_entertainment)
             (fears_depression_curiosity)
             (fears_anger_satiety)
             (craves_anger_onion)
             (craves_anger_tuna)
             (craves_anger_flounder)
             (craves_anger_cherry)
             (craves_anger_muffin)
             (craves_anger_ham)
             (fears_grief7_satiety)
             (fears_dread8_curiosity)
             (craves_grief7_tuna)
             (craves_grief7_onion)
             (craves_grief7_cherry)
             (craves_grief7_flounder)
             (craves_grief7_muffin)
             (fears_depression_entertainment)
             (fears_anger_stimulation)
             (fears_dread8_entertainment)
             (fears_grief7_stimulation)
             (fears_anger_curiosity)
             (fears_grief7_curiosity)
             (fears_depression1_satiety)
             (craves_depression1_onion)
             (craves_depression1_flounder)
             (craves_depression1_tuna)
             (craves_depression1_cherry)
             (craves_depression1_ham)
             (fears_anger_entertainment)
             (fears_depression1_stimulation)
             (fears_grief7_entertainment)
             (fears_jealousy2_satiety)
             (craves_jealousy2_beef)
             (fears_depression1_curiosity)
             (craves_jealousy2_tuna)
             (craves_jealousy2_onion)
             (craves_jealousy2_flounder)
             (craves_jealousy2_ham)
             (craves_jealousy2_muffin)
             (fears_depression1_entertainment)
             (fears_jealousy2_stimulation)
             (fears_jealousy2_curiosity)
             (fears_jealousy_satiety)
             (craves_jealousy_onion)
             (craves_jealousy_beef)
             (craves_jealousy_tuna)
             (craves_jealousy_muffin)
             (craves_jealousy_cherry)
             (craves_jealousy_ham)
             (fears_jealousy_stimulation)
             (fears_jealousy2_entertainment)
             (fears_anxiety_satiety)
             (craves_anxiety_beef)
             (fears_jealousy_curiosity)
             (craves_anxiety_tuna)
             (craves_anxiety_onion)
             (craves_anxiety_cherry)
             (craves_anxiety_ham)
             (craves_anxiety_muffin)
             (fears_jealousy_entertainment)
             (fears_anxiety_stimulation)
             (fears_anxiety_curiosity)
             (fears_hangover_satiety)
             (craves_hangover_beef)
             (craves_hangover_onion)
             (craves_hangover_flounder)
             (craves_hangover_cherry)
             (craves_hangover_muffin)
             (craves_hangover_ham)
             (fears_anxiety_entertainment)
             (fears_hangover_stimulation)
             (fears_hangover_curiosity)
             (fears_sciatica_satiety)
             (craves_sciatica_beef)
             (craves_sciatica_onion)
             (craves_sciatica_flounder)
             (craves_sciatica_cherry)
             (craves_sciatica_muffin)
             (craves_sciatica_ham)
             (fears_hangover_entertainment)
             (fears_sciatica_stimulation)
             (fears_sciatica_curiosity)
             (fears_laceration_satiety)
             (craves_laceration_beef)
             (craves_laceration_onion)
             (craves_laceration_flounder)
             (craves_laceration_cherry)
             (craves_laceration_muffin)
             (craves_laceration_ham)
             (fears_sciatica_entertainment)
             (fears_laceration_stimulation)
             (fears_laceration_curiosity)
             (fears_boils_satiety)
             (craves_boils_beef)
             (craves_boils_onion)
             (craves_boils_flounder)
             (craves_boils_cherry)
             (craves_boils_muffin)
             (craves_boils_ham)
             (fears_laceration_entertainment)
             (fears_boils_stimulation)
             (fears_boils_curiosity)
             (fears_angina_satiety)
             (craves_angina_beef)
             (craves_angina_tuna)
             (craves_angina_flounder)
             (craves_angina_cherry)
             (craves_angina_muffin)
             (craves_angina_ham)
             (fears_boils_entertainment)
             (fears_angina_stimulation)
             (fears_angina_curiosity)
             (fears_dread_satiety)
             (craves_dread_beef)
             (craves_dread_tuna)
             (craves_dread_flounder)
             (craves_dread_cherry)
             (craves_dread_muffin)
             (craves_dread_ham)
             (fears_angina_entertainment)
             (fears_dread_stimulation)
             (fears_dread_curiosity)
             (fears_loneliness_satiety)
             (craves_loneliness_beef)
             (craves_loneliness_tuna)
             (craves_loneliness_flounder)
             (craves_loneliness_cherry)
             (of)
 )
 (:derived (c_0_-1) (false))
(:derived (c_0_0) (or (v7_0) (and (c_0_-1) (not (v7_0)))))
(:derived (c_0_1) (and (c_0_0) (v7_1)))
(:derived (c_0_2) (and (c_0_1) (v7_2)))
(:derived (c_0_3) (and (c_0_2) (v7_3)))
(:derived (c_0_4) (and (c_0_3) (v7_4)))
(:derived (z_0_0) (or (and (not (v7_0)) (not (c_0_-1))) (and (c_0_-1) (v7_0))))
(:derived (z_0_1) (or (and (v7_1) (not (c_0_0))) (and (c_0_0) (not (v7_1)))))
(:derived (z_0_2) (or (and (v7_2) (not (c_0_1))) (and (c_0_1) (not (v7_2)))))
(:derived (z_0_3) (or (and (v7_3) (not (c_0_2))) (and (c_0_2) (not (v7_3)))))
(:derived (z_0_4) (or (and (v7_4) (not (c_0_3))) (and (c_0_3) (not (v7_4)))))
(:derived (c_1_-1) (false))
(:derived (c_1_0) (or (v4_0) (and (c_1_-1) (not (v4_0)))))
(:derived (c_1_1) (or (v4_1) (and (c_1_0) (not (v4_1)))))
(:derived (c_1_2) (or (v4_2) (and (c_1_1) (not (v4_2)))))
(:derived (c_1_3) (or (v4_3) (and (c_1_2) (not (v4_3)))))
(:derived (c_1_4) (or (v4_4) (and (c_1_3) (not (v4_4)))))
(:derived (z_1_0) (or (and (not (v4_0)) (not (c_1_-1))) (and (c_1_-1) (v4_0))))
(:derived (z_1_1) (or (and (not (v4_1)) (not (c_1_0))) (and (c_1_0) (v4_1))))
(:derived (z_1_2) (or (and (not (v4_2)) (not (c_1_1))) (and (c_1_1) (v4_2))))
(:derived (z_1_3) (or (and (not (v4_3)) (not (c_1_2))) (and (c_1_2) (v4_3))))
(:derived (z_1_4) (or (and (not (v4_4)) (not (c_1_3))) (and (c_1_3) (v4_4))))
(:derived (c_2_-1) (false))
(:derived (c_2_0) (or (v8_0) (and (c_2_-1) (not (v8_0)))))
(:derived (c_2_1) (and (c_2_0) (v8_1)))
(:derived (c_2_2) (and (c_2_1) (v8_2)))
(:derived (c_2_3) (and (c_2_2) (v8_3)))
(:derived (c_2_4) (and (c_2_3) (v8_4)))
(:derived (z_2_0) (or (and (not (v8_0)) (not (c_2_-1))) (and (c_2_-1) (v8_0))))
(:derived (z_2_1) (or (and (v8_1) (not (c_2_0))) (and (c_2_0) (not (v8_1)))))
(:derived (z_2_2) (or (and (v8_2) (not (c_2_1))) (and (c_2_1) (not (v8_2)))))
(:derived (z_2_3) (or (and (v8_3) (not (c_2_2))) (and (c_2_2) (not (v8_3)))))
(:derived (z_2_4) (or (and (v8_4) (not (c_2_3))) (and (c_2_3) (not (v8_4)))))
(:derived (c_3_-1) (false))
(:derived (c_3_0) (or (v5_0) (and (c_3_-1) (not (v5_0)))))
(:derived (c_3_1) (or (v5_1) (and (c_3_0) (not (v5_1)))))
(:derived (c_3_2) (or (v5_2) (and (c_3_1) (not (v5_2)))))
(:derived (c_3_3) (or (v5_3) (and (c_3_2) (not (v5_3)))))
(:derived (c_3_4) (or (v5_4) (and (c_3_3) (not (v5_4)))))
(:derived (z_3_0) (or (and (not (v5_0)) (not (c_3_-1))) (and (c_3_-1) (v5_0))))
(:derived (z_3_1) (or (and (not (v5_1)) (not (c_3_0))) (and (c_3_0) (v5_1))))
(:derived (z_3_2) (or (and (not (v5_2)) (not (c_3_1))) (and (c_3_1) (v5_2))))
(:derived (z_3_3) (or (and (not (v5_3)) (not (c_3_2))) (and (c_3_2) (v5_3))))
(:derived (z_3_4) (or (and (not (v5_4)) (not (c_3_3))) (and (c_3_3) (v5_4))))
(:derived (c_4_-1) (false))
(:derived (c_4_0) (or (v9_0) (and (c_4_-1) (not (v9_0)))))
(:derived (c_4_1) (and (c_4_0) (v9_1)))
(:derived (c_4_2) (and (c_4_1) (v9_2)))
(:derived (c_4_3) (and (c_4_2) (v9_3)))
(:derived (c_4_4) (and (c_4_3) (v9_4)))
(:derived (z_4_0) (or (and (not (v9_0)) (not (c_4_-1))) (and (c_4_-1) (v9_0))))
(:derived (z_4_1) (or (and (v9_1) (not (c_4_0))) (and (c_4_0) (not (v9_1)))))
(:derived (z_4_2) (or (and (v9_2) (not (c_4_1))) (and (c_4_1) (not (v9_2)))))
(:derived (z_4_3) (or (and (v9_3) (not (c_4_2))) (and (c_4_2) (not (v9_3)))))
(:derived (z_4_4) (or (and (v9_4) (not (c_4_3))) (and (c_4_3) (not (v9_4)))))
(:derived (c_5_-1) (false))
(:derived (c_5_0) (or (v6_0) (and (c_5_-1) (not (v6_0)))))
(:derived (c_5_1) (or (v6_1) (and (c_5_0) (not (v6_1)))))
(:derived (c_5_2) (or (v6_2) (and (c_5_1) (not (v6_2)))))
(:derived (c_5_3) (or (v6_3) (and (c_5_2) (not (v6_3)))))
(:derived (c_5_4) (or (v6_4) (and (c_5_3) (not (v6_4)))))
(:derived (z_5_0) (or (and (not (v6_0)) (not (c_5_-1))) (and (c_5_-1) (v6_0))))
(:derived (z_5_1) (or (and (not (v6_1)) (not (c_5_0))) (and (c_5_0) (v6_1))))
(:derived (z_5_2) (or (and (not (v6_2)) (not (c_5_1))) (and (c_5_1) (v6_2))))
(:derived (z_5_3) (or (and (not (v6_3)) (not (c_5_2))) (and (c_5_2) (v6_3))))
(:derived (z_5_4) (or (and (not (v6_4)) (not (c_5_3))) (and (c_5_3) (v6_4))))
(:derived (c_6_-1) (false))
(:derived (c_6_0) (or (v10_0) (and (c_6_-1) (not (v10_0)))))
(:derived (c_6_1) (and (c_6_0) (v10_1)))
(:derived (c_6_2) (and (c_6_1) (v10_2)))
(:derived (c_6_3) (and (c_6_2) (v10_3)))
(:derived (c_6_4) (and (c_6_3) (v10_4)))
(:derived (z_6_0) (or (and (not (v10_0)) (not (c_6_-1))) (and (c_6_-1) (v10_0))))
(:derived (z_6_1) (or (and (v10_1) (not (c_6_0))) (and (c_6_0) (not (v10_1)))))
(:derived (z_6_2) (or (and (v10_2) (not (c_6_1))) (and (c_6_1) (not (v10_2)))))
(:derived (z_6_3) (or (and (v10_3) (not (c_6_2))) (and (c_6_2) (not (v10_3)))))
(:derived (z_6_4) (or (and (v10_4) (not (c_6_3))) (and (c_6_3) (not (v10_4)))))
(:derived (c_7_-1) (false))
(:derived (c_7_0) (or (v7_0) (and (c_7_-1) (not (v7_0)))))
(:derived (c_7_1) (or (v7_1) (and (c_7_0) (not (v7_1)))))
(:derived (c_7_2) (or (v7_2) (and (c_7_1) (not (v7_2)))))
(:derived (c_7_3) (or (v7_3) (and (c_7_2) (not (v7_3)))))
(:derived (c_7_4) (or (v7_4) (and (c_7_3) (not (v7_4)))))
(:derived (z_7_0) (or (and (not (v7_0)) (not (c_7_-1))) (and (c_7_-1) (v7_0))))
(:derived (z_7_1) (or (and (not (v7_1)) (not (c_7_0))) (and (c_7_0) (v7_1))))
(:derived (z_7_2) (or (and (not (v7_2)) (not (c_7_1))) (and (c_7_1) (v7_2))))
(:derived (z_7_3) (or (and (not (v7_3)) (not (c_7_2))) (and (c_7_2) (v7_3))))
(:derived (z_7_4) (or (and (not (v7_4)) (not (c_7_3))) (and (c_7_3) (v7_4))))
(:derived (c_8_-1) (false))
(:derived (c_8_0) (or (v8_0) (and (c_8_-1) (not (v8_0)))))
(:derived (c_8_1) (or (v8_1) (and (c_8_0) (not (v8_1)))))
(:derived (c_8_2) (or (v8_2) (and (c_8_1) (not (v8_2)))))
(:derived (c_8_3) (or (v8_3) (and (c_8_2) (not (v8_3)))))
(:derived (c_8_4) (or (v8_4) (and (c_8_3) (not (v8_4)))))
(:derived (z_8_0) (or (and (not (v8_0)) (not (c_8_-1))) (and (c_8_-1) (v8_0))))
(:derived (z_8_1) (or (and (not (v8_1)) (not (c_8_0))) (and (c_8_0) (v8_1))))
(:derived (z_8_2) (or (and (not (v8_2)) (not (c_8_1))) (and (c_8_1) (v8_2))))
(:derived (z_8_3) (or (and (not (v8_3)) (not (c_8_2))) (and (c_8_2) (v8_3))))
(:derived (z_8_4) (or (and (not (v8_4)) (not (c_8_3))) (and (c_8_3) (v8_4))))
(:derived (c_9_-1) (false))
(:derived (c_9_0) (or (v9_0) (and (c_9_-1) (not (v9_0)))))
(:derived (c_9_1) (or (v9_1) (and (c_9_0) (not (v9_1)))))
(:derived (c_9_2) (or (v9_2) (and (c_9_1) (not (v9_2)))))
(:derived (c_9_3) (or (v9_3) (and (c_9_2) (not (v9_3)))))
(:derived (c_9_4) (or (v9_4) (and (c_9_3) (not (v9_4)))))
(:derived (z_9_0) (or (and (not (v9_0)) (not (c_9_-1))) (and (c_9_-1) (v9_0))))
(:derived (z_9_1) (or (and (not (v9_1)) (not (c_9_0))) (and (c_9_0) (v9_1))))
(:derived (z_9_2) (or (and (not (v9_2)) (not (c_9_1))) (and (c_9_1) (v9_2))))
(:derived (z_9_3) (or (and (not (v9_3)) (not (c_9_2))) (and (c_9_2) (v9_3))))
(:derived (z_9_4) (or (and (not (v9_4)) (not (c_9_3))) (and (c_9_3) (v9_4))))
(:derived (c_10_-1) (false))
(:derived (c_10_0) (or (v10_0) (and (c_10_-1) (not (v10_0)))))
(:derived (c_10_1) (or (v10_1) (and (c_10_0) (not (v10_1)))))
(:derived (c_10_2) (or (v10_2) (and (c_10_1) (not (v10_2)))))
(:derived (c_10_3) (or (v10_3) (and (c_10_2) (not (v10_3)))))
(:derived (c_10_4) (or (v10_4) (and (c_10_3) (not (v10_4)))))
(:derived (z_10_0) (or (and (not (v10_0)) (not (c_10_-1))) (and (c_10_-1) (v10_0))))
(:derived (z_10_1) (or (and (not (v10_1)) (not (c_10_0))) (and (c_10_0) (v10_1))))
(:derived (z_10_2) (or (and (not (v10_2)) (not (c_10_1))) (and (c_10_1) (v10_2))))
(:derived (z_10_3) (or (and (not (v10_3)) (not (c_10_2))) (and (c_10_2) (v10_3))))
(:derived (z_10_4) (or (and (not (v10_4)) (not (c_10_3))) (and (c_10_3) (v10_4))))
(:derived (c_11_-1) (false))
(:derived (c_11_0) (or (v0_0) (and (c_11_-1) (not (v0_0)))))
(:derived (c_11_1) (and (c_11_0) (v0_1)))
(:derived (c_11_2) (and (c_11_1) (v0_2)))
(:derived (c_11_3) (and (c_11_2) (v0_3)))
(:derived (c_11_4) (and (c_11_3) (v0_4)))
(:derived (z_11_0) (or (and (not (v0_0)) (not (c_11_-1))) (and (c_11_-1) (v0_0))))
(:derived (z_11_1) (or (and (v0_1) (not (c_11_0))) (and (c_11_0) (not (v0_1)))))
(:derived (z_11_2) (or (and (v0_2) (not (c_11_1))) (and (c_11_1) (not (v0_2)))))
(:derived (z_11_3) (or (and (v0_3) (not (c_11_2))) (and (c_11_2) (not (v0_3)))))
(:derived (z_11_4) (or (and (v0_4) (not (c_11_3))) (and (c_11_3) (not (v0_4)))))
(:derived (c_12_-1) (false))
(:derived (c_12_0) (or (v1_0) (and (c_12_-1) (not (v1_0)))))
(:derived (c_12_1) (and (c_12_0) (v1_1)))
(:derived (c_12_2) (and (c_12_1) (v1_2)))
(:derived (c_12_3) (and (c_12_2) (v1_3)))
(:derived (c_12_4) (and (c_12_3) (v1_4)))
(:derived (z_12_0) (or (and (not (v1_0)) (not (c_12_-1))) (and (c_12_-1) (v1_0))))
(:derived (z_12_1) (or (and (v1_1) (not (c_12_0))) (and (c_12_0) (not (v1_1)))))
(:derived (z_12_2) (or (and (v1_2) (not (c_12_1))) (and (c_12_1) (not (v1_2)))))
(:derived (z_12_3) (or (and (v1_3) (not (c_12_2))) (and (c_12_2) (not (v1_3)))))
(:derived (z_12_4) (or (and (v1_4) (not (c_12_3))) (and (c_12_3) (not (v1_4)))))
(:derived (c_13_-1) (false))
(:derived (c_13_0) (or (v2_0) (and (c_13_-1) (not (v2_0)))))
(:derived (c_13_1) (and (c_13_0) (v2_1)))
(:derived (c_13_2) (and (c_13_1) (v2_2)))
(:derived (c_13_3) (and (c_13_2) (v2_3)))
(:derived (c_13_4) (and (c_13_3) (v2_4)))
(:derived (z_13_0) (or (and (not (v2_0)) (not (c_13_-1))) (and (c_13_-1) (v2_0))))
(:derived (z_13_1) (or (and (v2_1) (not (c_13_0))) (and (c_13_0) (not (v2_1)))))
(:derived (z_13_2) (or (and (v2_2) (not (c_13_1))) (and (c_13_1) (not (v2_2)))))
(:derived (z_13_3) (or (and (v2_3) (not (c_13_2))) (and (c_13_2) (not (v2_3)))))
(:derived (z_13_4) (or (and (v2_4) (not (c_13_3))) (and (c_13_3) (not (v2_4)))))
(:derived (c_14_-1) (false))
(:derived (c_14_0) (or (v0_0) (and (c_14_-1) (not (v0_0)))))
(:derived (c_14_1) (or (v0_1) (and (c_14_0) (not (v0_1)))))
(:derived (c_14_2) (or (v0_2) (and (c_14_1) (not (v0_2)))))
(:derived (c_14_3) (or (v0_3) (and (c_14_2) (not (v0_3)))))
(:derived (c_14_4) (or (v0_4) (and (c_14_3) (not (v0_4)))))
(:derived (z_14_0) (or (and (not (v0_0)) (not (c_14_-1))) (and (c_14_-1) (v0_0))))
(:derived (z_14_1) (or (and (not (v0_1)) (not (c_14_0))) (and (c_14_0) (v0_1))))
(:derived (z_14_2) (or (and (not (v0_2)) (not (c_14_1))) (and (c_14_1) (v0_2))))
(:derived (z_14_3) (or (and (not (v0_3)) (not (c_14_2))) (and (c_14_2) (v0_3))))
(:derived (z_14_4) (or (and (not (v0_4)) (not (c_14_3))) (and (c_14_3) (v0_4))))
(:derived (c_15_-1) (false))
(:derived (c_15_0) (or (v3_0) (and (c_15_-1) (not (v3_0)))))
(:derived (c_15_1) (and (c_15_0) (v3_1)))
(:derived (c_15_2) (and (c_15_1) (v3_2)))
(:derived (c_15_3) (and (c_15_2) (v3_3)))
(:derived (c_15_4) (and (c_15_3) (v3_4)))
(:derived (z_15_0) (or (and (not (v3_0)) (not (c_15_-1))) (and (c_15_-1) (v3_0))))
(:derived (z_15_1) (or (and (v3_1) (not (c_15_0))) (and (c_15_0) (not (v3_1)))))
(:derived (z_15_2) (or (and (v3_2) (not (c_15_1))) (and (c_15_1) (not (v3_2)))))
(:derived (z_15_3) (or (and (v3_3) (not (c_15_2))) (and (c_15_2) (not (v3_3)))))
(:derived (z_15_4) (or (and (v3_4) (not (c_15_3))) (and (c_15_3) (not (v3_4)))))
(:derived (c_16_-1) (false))
(:derived (c_16_0) (or (v4_0) (and (c_16_-1) (not (v4_0)))))
(:derived (c_16_1) (and (c_16_0) (v4_1)))
(:derived (c_16_2) (and (c_16_1) (v4_2)))
(:derived (c_16_3) (and (c_16_2) (v4_3)))
(:derived (c_16_4) (and (c_16_3) (v4_4)))
(:derived (z_16_0) (or (and (not (v4_0)) (not (c_16_-1))) (and (c_16_-1) (v4_0))))
(:derived (z_16_1) (or (and (v4_1) (not (c_16_0))) (and (c_16_0) (not (v4_1)))))
(:derived (z_16_2) (or (and (v4_2) (not (c_16_1))) (and (c_16_1) (not (v4_2)))))
(:derived (z_16_3) (or (and (v4_3) (not (c_16_2))) (and (c_16_2) (not (v4_3)))))
(:derived (z_16_4) (or (and (v4_4) (not (c_16_3))) (and (c_16_3) (not (v4_4)))))
(:derived (c_17_-1) (false))
(:derived (c_17_0) (or (v1_0) (and (c_17_-1) (not (v1_0)))))
(:derived (c_17_1) (or (v1_1) (and (c_17_0) (not (v1_1)))))
(:derived (c_17_2) (or (v1_2) (and (c_17_1) (not (v1_2)))))
(:derived (c_17_3) (or (v1_3) (and (c_17_2) (not (v1_3)))))
(:derived (c_17_4) (or (v1_4) (and (c_17_3) (not (v1_4)))))
(:derived (z_17_0) (or (and (not (v1_0)) (not (c_17_-1))) (and (c_17_-1) (v1_0))))
(:derived (z_17_1) (or (and (not (v1_1)) (not (c_17_0))) (and (c_17_0) (v1_1))))
(:derived (z_17_2) (or (and (not (v1_2)) (not (c_17_1))) (and (c_17_1) (v1_2))))
(:derived (z_17_3) (or (and (not (v1_3)) (not (c_17_2))) (and (c_17_2) (v1_3))))
(:derived (z_17_4) (or (and (not (v1_4)) (not (c_17_3))) (and (c_17_3) (v1_4))))
(:derived (c_18_-1) (false))
(:derived (c_18_0) (or (v5_0) (and (c_18_-1) (not (v5_0)))))
(:derived (c_18_1) (and (c_18_0) (v5_1)))
(:derived (c_18_2) (and (c_18_1) (v5_2)))
(:derived (c_18_3) (and (c_18_2) (v5_3)))
(:derived (c_18_4) (and (c_18_3) (v5_4)))
(:derived (z_18_0) (or (and (not (v5_0)) (not (c_18_-1))) (and (c_18_-1) (v5_0))))
(:derived (z_18_1) (or (and (v5_1) (not (c_18_0))) (and (c_18_0) (not (v5_1)))))
(:derived (z_18_2) (or (and (v5_2) (not (c_18_1))) (and (c_18_1) (not (v5_2)))))
(:derived (z_18_3) (or (and (v5_3) (not (c_18_2))) (and (c_18_2) (not (v5_3)))))
(:derived (z_18_4) (or (and (v5_4) (not (c_18_3))) (and (c_18_3) (not (v5_4)))))
(:derived (c_19_-1) (false))
(:derived (c_19_0) (or (v2_0) (and (c_19_-1) (not (v2_0)))))
(:derived (c_19_1) (or (v2_1) (and (c_19_0) (not (v2_1)))))
(:derived (c_19_2) (or (v2_2) (and (c_19_1) (not (v2_2)))))
(:derived (c_19_3) (or (v2_3) (and (c_19_2) (not (v2_3)))))
(:derived (c_19_4) (or (v2_4) (and (c_19_3) (not (v2_4)))))
(:derived (z_19_0) (or (and (not (v2_0)) (not (c_19_-1))) (and (c_19_-1) (v2_0))))
(:derived (z_19_1) (or (and (not (v2_1)) (not (c_19_0))) (and (c_19_0) (v2_1))))
(:derived (z_19_2) (or (and (not (v2_2)) (not (c_19_1))) (and (c_19_1) (v2_2))))
(:derived (z_19_3) (or (and (not (v2_3)) (not (c_19_2))) (and (c_19_2) (v2_3))))
(:derived (z_19_4) (or (and (not (v2_4)) (not (c_19_3))) (and (c_19_3) (v2_4))))
(:derived (c_20_-1) (false))
(:derived (c_20_0) (or (v6_0) (and (c_20_-1) (not (v6_0)))))
(:derived (c_20_1) (and (c_20_0) (v6_1)))
(:derived (c_20_2) (and (c_20_1) (v6_2)))
(:derived (c_20_3) (and (c_20_2) (v6_3)))
(:derived (c_20_4) (and (c_20_3) (v6_4)))
(:derived (z_20_0) (or (and (not (v6_0)) (not (c_20_-1))) (and (c_20_-1) (v6_0))))
(:derived (z_20_1) (or (and (v6_1) (not (c_20_0))) (and (c_20_0) (not (v6_1)))))
(:derived (z_20_2) (or (and (v6_2) (not (c_20_1))) (and (c_20_1) (not (v6_2)))))
(:derived (z_20_3) (or (and (v6_3) (not (c_20_2))) (and (c_20_2) (not (v6_3)))))
(:derived (z_20_4) (or (and (v6_4) (not (c_20_3))) (and (c_20_3) (not (v6_4)))))
(:derived (c_21_-1) (false))
(:derived (c_21_0) (or (v3_0) (and (c_21_-1) (not (v3_0)))))
(:derived (c_21_1) (or (v3_1) (and (c_21_0) (not (v3_1)))))
(:derived (c_21_2) (or (v3_2) (and (c_21_1) (not (v3_2)))))
(:derived (c_21_3) (or (v3_3) (and (c_21_2) (not (v3_3)))))
(:derived (c_21_4) (or (v3_4) (and (c_21_3) (not (v3_4)))))
(:derived (z_21_0) (or (and (not (v3_0)) (not (c_21_-1))) (and (c_21_-1) (v3_0))))
(:derived (z_21_1) (or (and (not (v3_1)) (not (c_21_0))) (and (c_21_0) (v3_1))))
(:derived (z_21_2) (or (and (not (v3_2)) (not (c_21_1))) (and (c_21_1) (v3_2))))
(:derived (z_21_3) (or (and (not (v3_3)) (not (c_21_2))) (and (c_21_2) (v3_3))))
(:derived (z_21_4) (or (and (not (v3_4)) (not (c_21_3))) (and (c_21_3) (v3_4))))
(:action overcome__dread8__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_dread8_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_dread8_ham)) (fears_dread8_entertainment)))
 (:action overcome__boils4__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_boils4_ham) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_boils4_ham)) (fears_boils4_entertainment)))
 (:action feast__satiety__onion__beef
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (not (craves_satiety_onion)) (craves_satiety_beef)))
 (:action feast__satiety__onion__cherry
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (not (craves_satiety_onion)) (craves_satiety_cherry)))
 (:action feast__satiety__onion__muffin
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (not (craves_satiety_onion)) (craves_satiety_muffin)))
 (:action feast__curiosity__cherry__onion
  :parameters ()
  :precondition (and (not (v2_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (craves_curiosity_onion) (not (craves_curiosity_cherry))))
 (:action feast__curiosity__cherry__beef
  :parameters ()
  :precondition (and (not (v2_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (craves_curiosity_beef) (not (craves_curiosity_cherry))))
 (:action feast__curiosity__cherry__flounder
  :parameters ()
  :precondition (and (not (v2_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (craves_curiosity_flounder) (not (craves_curiosity_cherry))))
 (:action overcome__dread__satiety__onion
  :parameters ()
  :precondition (and (craves_dread_onion) (craves_satiety_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread_onion)) (fears_dread_satiety)))
 (:action overcome__anxiety__stimulation__flounder
  :parameters ()
  :precondition (and (craves_anxiety_flounder) (not (v4_4)) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_anxiety_flounder)) (fears_anxiety_stimulation)))
 (:action overcome__grief7__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_grief7_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief7_entertainment) (not (craves_grief7_ham))))
 (:action overcome__jealousy2__curiosity__cherry
  :parameters ()
  :precondition (and (craves_jealousy2_cherry) (not (v5_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy2_cherry)) (fears_jealousy2_curiosity)))
 (:action drink__beef__muffin
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_6_0) (v10_0)) (when (not (z_6_0)) (not (v10_0))) (when (z_6_1) (v10_1)) (when (not (z_6_1)) (not (v10_1))) (when (z_6_2) (v10_2)) (when (not (z_6_2)) (not (v10_2))) (when (z_6_3) (v10_3)) (when (not (z_6_3)) (not (v10_3))) (when (z_6_4) (v10_4)) (when (not (z_6_4)) (not (v10_4))) (when (and (not (v10_4)) (z_6_4)) (of)) (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of))))
 (:action drink__beef__cherry
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_13_0) (v2_0)) (when (not (z_13_0)) (not (v2_0))) (when (z_13_1) (v2_1)) (when (not (z_13_1)) (not (v2_1))) (when (z_13_2) (v2_2)) (when (not (z_13_2)) (not (v2_2))) (when (z_13_3) (v2_3)) (when (not (z_13_3)) (not (v2_3))) (when (z_13_4) (v2_4)) (when (not (z_13_4)) (not (v2_4))) (when (and (not (v2_4)) (z_13_4)) (of)) (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of))))
 (:action drink__beef__ham
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_4_0) (v9_0)) (when (not (z_4_0)) (not (v9_0))) (when (z_4_1) (v9_1)) (when (not (z_4_1)) (not (v9_1))) (when (z_4_2) (v9_2)) (when (not (z_4_2)) (not (v9_2))) (when (z_4_3) (v9_3)) (when (not (z_4_3)) (not (v9_3))) (when (z_4_4) (v9_4)) (when (not (z_4_4)) (not (v9_4))) (when (and (not (v9_4)) (z_4_4)) (of)) (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of))))
 (:action drink__beef__onion
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_12_0) (v1_0)) (when (not (z_12_0)) (not (v1_0))) (when (z_12_1) (v1_1)) (when (not (z_12_1)) (not (v1_1))) (when (z_12_2) (v1_2)) (when (not (z_12_2)) (not (v1_2))) (when (z_12_3) (v1_3)) (when (not (z_12_3)) (not (v1_3))) (when (z_12_4) (v1_4)) (when (not (z_12_4)) (not (v1_4))) (when (and (not (v1_4)) (z_12_4)) (of)) (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of))))
 (:action drink__beef__flounder
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (when (z_2_0) (v8_0)) (when (not (z_2_0)) (not (v8_0))) (when (z_2_1) (v8_1)) (when (not (z_2_1)) (not (v8_1))) (when (z_2_2) (v8_2)) (when (not (z_2_2)) (not (v8_2))) (when (z_2_3) (v8_3)) (when (not (z_2_3)) (not (v8_3))) (when (z_2_4) (v8_4)) (when (not (z_2_4)) (not (v8_4))) (when (and (not (v8_4)) (z_2_4)) (of))))
 (:action drink__beef__tuna
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_0_0) (v7_0)) (when (not (z_0_0)) (not (v7_0))) (when (z_0_1) (v7_1)) (when (not (z_0_1)) (not (v7_1))) (when (z_0_2) (v7_2)) (when (not (z_0_2)) (not (v7_2))) (when (z_0_3) (v7_3)) (when (not (z_0_3)) (not (v7_3))) (when (z_0_4) (v7_4)) (when (not (z_0_4)) (not (v7_4))) (when (and (not (v7_4)) (z_0_4)) (of)) (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of))))
 (:action overcome__prostatitis3__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_prostatitis3_ham) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_prostatitis3_ham)) (fears_prostatitis3_entertainment)))
 (:action drink__onion__ham
  :parameters ()
  :precondition (and (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v9_0)) (when (not (z_4_0)) (not (v9_0))) (when (z_4_1) (v9_1)) (when (not (z_4_1)) (not (v9_1))) (when (z_4_2) (v9_2)) (when (not (z_4_2)) (not (v9_2))) (when (z_4_3) (v9_3)) (when (not (z_4_3)) (not (v9_3))) (when (z_4_4) (v9_4)) (when (not (z_4_4)) (not (v9_4))) (when (and (not (v9_4)) (z_4_4)) (of)) (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of))))
 (:action drink__onion__muffin
  :parameters ()
  :precondition (and (not (v1_4)) (not (of)))
  :effect (and (when (z_6_0) (v10_0)) (when (not (z_6_0)) (not (v10_0))) (when (z_6_1) (v10_1)) (when (not (z_6_1)) (not (v10_1))) (when (z_6_2) (v10_2)) (when (not (z_6_2)) (not (v10_2))) (when (z_6_3) (v10_3)) (when (not (z_6_3)) (not (v10_3))) (when (z_6_4) (v10_4)) (when (not (z_6_4)) (not (v10_4))) (when (and (not (v10_4)) (z_6_4)) (of)) (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of))))
 (:action drink__onion__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v7_0)) (when (not (z_0_0)) (not (v7_0))) (when (z_0_1) (v7_1)) (when (not (z_0_1)) (not (v7_1))) (when (z_0_2) (v7_2)) (when (not (z_0_2)) (not (v7_2))) (when (z_0_3) (v7_3)) (when (not (z_0_3)) (not (v7_3))) (when (z_0_4) (v7_4)) (when (not (z_0_4)) (not (v7_4))) (when (and (not (v7_4)) (z_0_4)) (of)) (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of))))
 (:action drink__onion__cherry
  :parameters ()
  :precondition (and (not (v1_4)) (not (of)))
  :effect (and (when (z_13_0) (v2_0)) (when (not (z_13_0)) (not (v2_0))) (when (z_13_1) (v2_1)) (when (not (z_13_1)) (not (v2_1))) (when (z_13_2) (v2_2)) (when (not (z_13_2)) (not (v2_2))) (when (z_13_3) (v2_3)) (when (not (z_13_3)) (not (v2_3))) (when (z_13_4) (v2_4)) (when (not (z_13_4)) (not (v2_4))) (when (and (not (v2_4)) (z_13_4)) (of)) (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of))))
 (:action drink__onion__flounder
  :parameters ()
  :precondition (and (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (when (z_2_0) (v8_0)) (when (not (z_2_0)) (not (v8_0))) (when (z_2_1) (v8_1)) (when (not (z_2_1)) (not (v8_1))) (when (z_2_2) (v8_2)) (when (not (z_2_2)) (not (v8_2))) (when (z_2_3) (v8_3)) (when (not (z_2_3)) (not (v8_3))) (when (z_2_4) (v8_4)) (when (not (z_2_4)) (not (v8_4))) (when (and (not (v8_4)) (z_2_4)) (of))))
 (:action drink__onion__beef
  :parameters ()
  :precondition (and (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (when (z_20_0) (v6_0)) (when (not (z_20_0)) (not (v6_0))) (when (z_20_1) (v6_1)) (when (not (z_20_1)) (not (v6_1))) (when (z_20_2) (v6_2)) (when (not (z_20_2)) (not (v6_2))) (when (z_20_3) (v6_3)) (when (not (z_20_3)) (not (v6_3))) (when (z_20_4) (v6_4)) (when (not (z_20_4)) (not (v6_4))) (when (and (not (v6_4)) (z_20_4)) (of))))
 (:action drink__tuna__ham
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (when (z_4_0) (v9_0)) (when (not (z_4_0)) (not (v9_0))) (when (z_4_1) (v9_1)) (when (not (z_4_1)) (not (v9_1))) (when (z_4_2) (v9_2)) (when (not (z_4_2)) (not (v9_2))) (when (z_4_3) (v9_3)) (when (not (z_4_3)) (not (v9_3))) (when (z_4_4) (v9_4)) (when (not (z_4_4)) (not (v9_4))) (when (and (not (v9_4)) (z_4_4)) (of))))
 (:action drink__tuna__flounder
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (when (z_2_0) (v8_0)) (when (not (z_2_0)) (not (v8_0))) (when (z_2_1) (v8_1)) (when (not (z_2_1)) (not (v8_1))) (when (z_2_2) (v8_2)) (when (not (z_2_2)) (not (v8_2))) (when (z_2_3) (v8_3)) (when (not (z_2_3)) (not (v8_3))) (when (z_2_4) (v8_4)) (when (not (z_2_4)) (not (v8_4))) (when (and (not (v8_4)) (z_2_4)) (of))))
 (:action drink__tuna__muffin
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_6_0) (v10_0)) (when (not (z_6_0)) (not (v10_0))) (when (z_6_1) (v10_1)) (when (not (z_6_1)) (not (v10_1))) (when (z_6_2) (v10_2)) (when (not (z_6_2)) (not (v10_2))) (when (z_6_3) (v10_3)) (when (not (z_6_3)) (not (v10_3))) (when (z_6_4) (v10_4)) (when (not (z_6_4)) (not (v10_4))) (when (and (not (v10_4)) (z_6_4)) (of)) (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of))))
 (:action drink__tuna__cherry
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_13_0) (v2_0)) (when (not (z_13_0)) (not (v2_0))) (when (z_13_1) (v2_1)) (when (not (z_13_1)) (not (v2_1))) (when (z_13_2) (v2_2)) (when (not (z_13_2)) (not (v2_2))) (when (z_13_3) (v2_3)) (when (not (z_13_3)) (not (v2_3))) (when (z_13_4) (v2_4)) (when (not (z_13_4)) (not (v2_4))) (when (and (not (v2_4)) (z_13_4)) (of)) (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of))))
 (:action drink__tuna__onion
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_12_0) (v1_0)) (when (not (z_12_0)) (not (v1_0))) (when (z_12_1) (v1_1)) (when (not (z_12_1)) (not (v1_1))) (when (z_12_2) (v1_2)) (when (not (z_12_2)) (not (v1_2))) (when (z_12_3) (v1_3)) (when (not (z_12_3)) (not (v1_3))) (when (z_12_4) (v1_4)) (when (not (z_12_4)) (not (v1_4))) (when (and (not (v1_4)) (z_12_4)) (of)) (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of))))
 (:action drink__tuna__beef
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (when (z_20_0) (v6_0)) (when (not (z_20_0)) (not (v6_0))) (when (z_20_1) (v6_1)) (when (not (z_20_1)) (not (v6_1))) (when (z_20_2) (v6_2)) (when (not (z_20_2)) (not (v6_2))) (when (z_20_3) (v6_3)) (when (not (z_20_3)) (not (v6_3))) (when (z_20_4) (v6_4)) (when (not (z_20_4)) (not (v6_4))) (when (and (not (v6_4)) (z_20_4)) (of))))
 (:action drink__flounder__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v2_0)) (when (not (z_13_0)) (not (v2_0))) (when (z_13_1) (v2_1)) (when (not (z_13_1)) (not (v2_1))) (when (z_13_2) (v2_2)) (when (not (z_13_2)) (not (v2_2))) (when (z_13_3) (v2_3)) (when (not (z_13_3)) (not (v2_3))) (when (z_13_4) (v2_4)) (when (not (z_13_4)) (not (v2_4))) (when (and (not (v2_4)) (z_13_4)) (of)) (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of))))
 (:action drink__flounder__ham
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_4_0) (v9_0)) (when (not (z_4_0)) (not (v9_0))) (when (z_4_1) (v9_1)) (when (not (z_4_1)) (not (v9_1))) (when (z_4_2) (v9_2)) (when (not (z_4_2)) (not (v9_2))) (when (z_4_3) (v9_3)) (when (not (z_4_3)) (not (v9_3))) (when (z_4_4) (v9_4)) (when (not (z_4_4)) (not (v9_4))) (when (and (not (v9_4)) (z_4_4)) (of)) (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of))))
 (:action drink__flounder__muffin
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_6_0) (v10_0)) (when (not (z_6_0)) (not (v10_0))) (when (z_6_1) (v10_1)) (when (not (z_6_1)) (not (v10_1))) (when (z_6_2) (v10_2)) (when (not (z_6_2)) (not (v10_2))) (when (z_6_3) (v10_3)) (when (not (z_6_3)) (not (v10_3))) (when (z_6_4) (v10_4)) (when (not (z_6_4)) (not (v10_4))) (when (and (not (v10_4)) (z_6_4)) (of)) (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of))))
 (:action drink__flounder__beef
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_20_0) (v6_0)) (when (not (z_20_0)) (not (v6_0))) (when (z_20_1) (v6_1)) (when (not (z_20_1)) (not (v6_1))) (when (z_20_2) (v6_2)) (when (not (z_20_2)) (not (v6_2))) (when (z_20_3) (v6_3)) (when (not (z_20_3)) (not (v6_3))) (when (z_20_4) (v6_4)) (when (not (z_20_4)) (not (v6_4))) (when (and (not (v6_4)) (z_20_4)) (of)) (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of))))
 (:action drink__flounder__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_0_0) (v7_0)) (when (not (z_0_0)) (not (v7_0))) (when (z_0_1) (v7_1)) (when (not (z_0_1)) (not (v7_1))) (when (z_0_2) (v7_2)) (when (not (z_0_2)) (not (v7_2))) (when (z_0_3) (v7_3)) (when (not (z_0_3)) (not (v7_3))) (when (z_0_4) (v7_4)) (when (not (z_0_4)) (not (v7_4))) (when (and (not (v7_4)) (z_0_4)) (of)) (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of))))
 (:action drink__flounder__onion
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_12_0) (v1_0)) (when (not (z_12_0)) (not (v1_0))) (when (z_12_1) (v1_1)) (when (not (z_12_1)) (not (v1_1))) (when (z_12_2) (v1_2)) (when (not (z_12_2)) (not (v1_2))) (when (z_12_3) (v1_3)) (when (not (z_12_3)) (not (v1_3))) (when (z_12_4) (v1_4)) (when (not (z_12_4)) (not (v1_4))) (when (and (not (v1_4)) (z_12_4)) (of)) (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of))))
 (:action feast__stimulation__flounder__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of)) (craves_stimulation_cherry) (not (craves_stimulation_flounder))))
 (:action feast__stimulation__flounder__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of)) (craves_stimulation_tuna) (not (craves_stimulation_flounder))))
 (:action feast__entertainment__ham__tuna
  :parameters ()
  :precondition (and (craves_entertainment_ham) (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (craves_entertainment_tuna) (not (craves_entertainment_ham))))
 (:action feast__entertainment__ham__muffin
  :parameters ()
  :precondition (and (craves_entertainment_ham) (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (craves_entertainment_muffin) (not (craves_entertainment_ham))))
 (:action drink__cherry__muffin
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_6_0) (v10_0)) (when (not (z_6_0)) (not (v10_0))) (when (z_6_1) (v10_1)) (when (not (z_6_1)) (not (v10_1))) (when (z_6_2) (v10_2)) (when (not (z_6_2)) (not (v10_2))) (when (z_6_3) (v10_3)) (when (not (z_6_3)) (not (v10_3))) (when (z_6_4) (v10_4)) (when (not (z_6_4)) (not (v10_4))) (when (and (not (v10_4)) (z_6_4)) (of)) (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of))))
 (:action drink__cherry__ham
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (when (z_4_0) (v9_0)) (when (not (z_4_0)) (not (v9_0))) (when (z_4_1) (v9_1)) (when (not (z_4_1)) (not (v9_1))) (when (z_4_2) (v9_2)) (when (not (z_4_2)) (not (v9_2))) (when (z_4_3) (v9_3)) (when (not (z_4_3)) (not (v9_3))) (when (z_4_4) (v9_4)) (when (not (z_4_4)) (not (v9_4))) (when (and (not (v9_4)) (z_4_4)) (of))))
 (:action drink__cherry__onion
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (when (z_12_0) (v1_0)) (when (not (z_12_0)) (not (v1_0))) (when (z_12_1) (v1_1)) (when (not (z_12_1)) (not (v1_1))) (when (z_12_2) (v1_2)) (when (not (z_12_2)) (not (v1_2))) (when (z_12_3) (v1_3)) (when (not (z_12_3)) (not (v1_3))) (when (z_12_4) (v1_4)) (when (not (z_12_4)) (not (v1_4))) (when (and (not (v1_4)) (z_12_4)) (of))))
 (:action drink__cherry__beef
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (when (z_20_0) (v6_0)) (when (not (z_20_0)) (not (v6_0))) (when (z_20_1) (v6_1)) (when (not (z_20_1)) (not (v6_1))) (when (z_20_2) (v6_2)) (when (not (z_20_2)) (not (v6_2))) (when (z_20_3) (v6_3)) (when (not (z_20_3)) (not (v6_3))) (when (z_20_4) (v6_4)) (when (not (z_20_4)) (not (v6_4))) (when (and (not (v6_4)) (z_20_4)) (of))))
 (:action drink__cherry__flounder
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (when (z_2_0) (v8_0)) (when (not (z_2_0)) (not (v8_0))) (when (z_2_1) (v8_1)) (when (not (z_2_1)) (not (v8_1))) (when (z_2_2) (v8_2)) (when (not (z_2_2)) (not (v8_2))) (when (z_2_3) (v8_3)) (when (not (z_2_3)) (not (v8_3))) (when (z_2_4) (v8_4)) (when (not (z_2_4)) (not (v8_4))) (when (and (not (v8_4)) (z_2_4)) (of))))
 (:action drink__cherry__tuna
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (when (z_0_0) (v7_0)) (when (not (z_0_0)) (not (v7_0))) (when (z_0_1) (v7_1)) (when (not (z_0_1)) (not (v7_1))) (when (z_0_2) (v7_2)) (when (not (z_0_2)) (not (v7_2))) (when (z_0_3) (v7_3)) (when (not (z_0_3)) (not (v7_3))) (when (z_0_4) (v7_4)) (when (not (z_0_4)) (not (v7_4))) (when (and (not (v7_4)) (z_0_4)) (of))))
 (:action drink__muffin__beef
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (when (z_20_0) (v6_0)) (when (not (z_20_0)) (not (v6_0))) (when (z_20_1) (v6_1)) (when (not (z_20_1)) (not (v6_1))) (when (z_20_2) (v6_2)) (when (not (z_20_2)) (not (v6_2))) (when (z_20_3) (v6_3)) (when (not (z_20_3)) (not (v6_3))) (when (z_20_4) (v6_4)) (when (not (z_20_4)) (not (v6_4))) (when (and (not (v6_4)) (z_20_4)) (of))))
 (:action drink__muffin__ham
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (when (z_4_0) (v9_0)) (when (not (z_4_0)) (not (v9_0))) (when (z_4_1) (v9_1)) (when (not (z_4_1)) (not (v9_1))) (when (z_4_2) (v9_2)) (when (not (z_4_2)) (not (v9_2))) (when (z_4_3) (v9_3)) (when (not (z_4_3)) (not (v9_3))) (when (z_4_4) (v9_4)) (when (not (z_4_4)) (not (v9_4))) (when (and (not (v9_4)) (z_4_4)) (of))))
 (:action drink__muffin__tuna
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (when (z_0_0) (v7_0)) (when (not (z_0_0)) (not (v7_0))) (when (z_0_1) (v7_1)) (when (not (z_0_1)) (not (v7_1))) (when (z_0_2) (v7_2)) (when (not (z_0_2)) (not (v7_2))) (when (z_0_3) (v7_3)) (when (not (z_0_3)) (not (v7_3))) (when (z_0_4) (v7_4)) (when (not (z_0_4)) (not (v7_4))) (when (and (not (v7_4)) (z_0_4)) (of))))
 (:action drink__muffin__onion
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (when (z_12_0) (v1_0)) (when (not (z_12_0)) (not (v1_0))) (when (z_12_1) (v1_1)) (when (not (z_12_1)) (not (v1_1))) (when (z_12_2) (v1_2)) (when (not (z_12_2)) (not (v1_2))) (when (z_12_3) (v1_3)) (when (not (z_12_3)) (not (v1_3))) (when (z_12_4) (v1_4)) (when (not (z_12_4)) (not (v1_4))) (when (and (not (v1_4)) (z_12_4)) (of))))
 (:action drink__muffin__cherry
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_13_0) (v2_0)) (when (not (z_13_0)) (not (v2_0))) (when (z_13_1) (v2_1)) (when (not (z_13_1)) (not (v2_1))) (when (z_13_2) (v2_2)) (when (not (z_13_2)) (not (v2_2))) (when (z_13_3) (v2_3)) (when (not (z_13_3)) (not (v2_3))) (when (z_13_4) (v2_4)) (when (not (z_13_4)) (not (v2_4))) (when (and (not (v2_4)) (z_13_4)) (of)) (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of))))
 (:action drink__muffin__flounder
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (when (z_2_0) (v8_0)) (when (not (z_2_0)) (not (v8_0))) (when (z_2_1) (v8_1)) (when (not (z_2_1)) (not (v8_1))) (when (z_2_2) (v8_2)) (when (not (z_2_2)) (not (v8_2))) (when (z_2_3) (v8_3)) (when (not (z_2_3)) (not (v8_3))) (when (z_2_4) (v8_4)) (when (not (z_2_4)) (not (v8_4))) (when (and (not (v8_4)) (z_2_4)) (of))))
 (:action drink__ham__onion
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_12_0) (v1_0)) (when (not (z_12_0)) (not (v1_0))) (when (z_12_1) (v1_1)) (when (not (z_12_1)) (not (v1_1))) (when (z_12_2) (v1_2)) (when (not (z_12_2)) (not (v1_2))) (when (z_12_3) (v1_3)) (when (not (z_12_3)) (not (v1_3))) (when (z_12_4) (v1_4)) (when (not (z_12_4)) (not (v1_4))) (when (and (not (v1_4)) (z_12_4)) (of)) (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of))))
 (:action drink__ham__beef
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (when (z_20_0) (v6_0)) (when (not (z_20_0)) (not (v6_0))) (when (z_20_1) (v6_1)) (when (not (z_20_1)) (not (v6_1))) (when (z_20_2) (v6_2)) (when (not (z_20_2)) (not (v6_2))) (when (z_20_3) (v6_3)) (when (not (z_20_3)) (not (v6_3))) (when (z_20_4) (v6_4)) (when (not (z_20_4)) (not (v6_4))) (when (and (not (v6_4)) (z_20_4)) (of))))
 (:action overcome__loneliness__satiety__onion
  :parameters ()
  :precondition (and (craves_loneliness_onion) (craves_satiety_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_loneliness_onion)) (fears_loneliness_satiety)))
 (:action drink__ham__flounder
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (when (z_2_0) (v8_0)) (when (not (z_2_0)) (not (v8_0))) (when (z_2_1) (v8_1)) (when (not (z_2_1)) (not (v8_1))) (when (z_2_2) (v8_2)) (when (not (z_2_2)) (not (v8_2))) (when (z_2_3) (v8_3)) (when (not (z_2_3)) (not (v8_3))) (when (z_2_4) (v8_4)) (when (not (z_2_4)) (not (v8_4))) (when (and (not (v8_4)) (z_2_4)) (of))))
 (:action drink__ham__tuna
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_0_0) (v7_0)) (when (not (z_0_0)) (not (v7_0))) (when (z_0_1) (v7_1)) (when (not (z_0_1)) (not (v7_1))) (when (z_0_2) (v7_2)) (when (not (z_0_2)) (not (v7_2))) (when (z_0_3) (v7_3)) (when (not (z_0_3)) (not (v7_3))) (when (z_0_4) (v7_4)) (when (not (z_0_4)) (not (v7_4))) (when (and (not (v7_4)) (z_0_4)) (of)) (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of))))
 (:action drink__ham__muffin
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_6_0) (v10_0)) (when (not (z_6_0)) (not (v10_0))) (when (z_6_1) (v10_1)) (when (not (z_6_1)) (not (v10_1))) (when (z_6_2) (v10_2)) (when (not (z_6_2)) (not (v10_2))) (when (z_6_3) (v10_3)) (when (not (z_6_3)) (not (v10_3))) (when (z_6_4) (v10_4)) (when (not (z_6_4)) (not (v10_4))) (when (and (not (v10_4)) (z_6_4)) (of)) (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of))))
 (:action drink__ham__cherry
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v2_0)) (when (not (z_13_0)) (not (v2_0))) (when (z_13_1) (v2_1)) (when (not (z_13_1)) (not (v2_1))) (when (z_13_2) (v2_2)) (when (not (z_13_2)) (not (v2_2))) (when (z_13_3) (v2_3)) (when (not (z_13_3)) (not (v2_3))) (when (z_13_4) (v2_4)) (when (not (z_13_4)) (not (v2_4))) (when (and (not (v2_4)) (z_13_4)) (of)) (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of))))
 (:action overcome__angina__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (craves_angina_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_angina_onion)) (fears_angina_satiety)))
 (:action overcome__jealousy__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_jealousy_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_jealousy_flounder)) (fears_jealousy_stimulation)))
 (:action overcome__anger__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (not (v3_4)) (craves_anger_beef) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_anger_satiety) (not (craves_anger_beef))))
 (:action succumb__angina__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_angina_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_angina_muffin) (not (fears_angina_satiety))))
 (:action overcome__depression1__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (craves_depression1_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression1_muffin)) (fears_depression1_satiety)))
 (:action succumb__angina__satiety__beef
  :parameters ()
  :precondition (and (fears_angina_satiety) (craves_satiety_beef) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_angina_satiety)) (craves_angina_beef)))
 (:action succumb__angina__satiety__onion
  :parameters ()
  :precondition (and (fears_angina_satiety) (craves_satiety_onion) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_angina_onion) (not (fears_angina_satiety))))
 (:action succumb__angina__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_angina_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_angina_cherry) (not (fears_angina_satiety))))
 (:action overcome__anger__curiosity__beef
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_beef) (craves_anger_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_anger_curiosity) (not (craves_anger_beef))))
 (:action feast__curiosity__onion__beef
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (not (craves_curiosity_onion)) (craves_curiosity_beef)))
 (:action feast__curiosity__onion__muffin
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (not (craves_curiosity_onion)) (craves_curiosity_muffin)))
 (:action feast__curiosity__onion__cherry
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (not (craves_curiosity_onion)) (craves_curiosity_cherry)))
 (:action overcome__prostatitis__satiety__beef
  :parameters ()
  :precondition (and (craves_prostatitis_beef) (craves_satiety_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_prostatitis_beef)) (fears_prostatitis_satiety)))
 (:action feast__curiosity__beef__onion
  :parameters ()
  :precondition (and (craves_curiosity_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (craves_curiosity_onion) (not (craves_curiosity_beef))))
 (:action feast__curiosity__beef__cherry
  :parameters ()
  :precondition (and (craves_curiosity_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (not (craves_curiosity_beef)) (craves_curiosity_cherry)))
 (:action feast__curiosity__beef__tuna
  :parameters ()
  :precondition (and (craves_curiosity_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (craves_curiosity_tuna) (not (craves_curiosity_beef))))
 (:action succumb__loneliness__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_loneliness_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_loneliness_satiety)) (craves_loneliness_beef)))
 (:action succumb__loneliness__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_loneliness_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_loneliness_satiety)) (craves_loneliness_onion)))
 (:action succumb__loneliness__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_loneliness_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_loneliness_satiety)) (craves_loneliness_cherry)))
 (:action succumb__loneliness__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_loneliness_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_loneliness_satiety)) (craves_loneliness_muffin)))
 (:action feast__curiosity__flounder__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of)) (craves_curiosity_cherry) (not (craves_curiosity_flounder))))
 (:action feast__curiosity__flounder__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of)) (craves_curiosity_tuna) (not (craves_curiosity_flounder))))
 (:action overcome__prostatitis__curiosity__beef
  :parameters ()
  :precondition (and (craves_prostatitis_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_prostatitis_beef)) (fears_prostatitis_curiosity)))
 (:action overcome__abrasion__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (not (v3_4)) (craves_abrasion_beef) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_abrasion_satiety) (not (craves_abrasion_beef))))
 (:action feast__satiety__beef__tuna
  :parameters ()
  :precondition (and (craves_satiety_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (craves_satiety_tuna) (not (craves_satiety_beef))))
 (:action feast__satiety__beef__onion
  :parameters ()
  :precondition (and (craves_satiety_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (craves_satiety_onion) (not (craves_satiety_beef))))
 (:action feast__satiety__beef__cherry
  :parameters ()
  :precondition (and (craves_satiety_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (craves_satiety_cherry) (not (craves_satiety_beef))))
 (:action overcome__abrasion__curiosity__beef
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_beef) (craves_abrasion_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_beef))))
 (:action feast__satiety__muffin__tuna
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_satiety_tuna) (not (craves_satiety_muffin))))
 (:action feast__satiety__muffin__onion
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_satiety_onion) (not (craves_satiety_muffin))))
 (:action feast__satiety__muffin__ham
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_satiety_ham) (not (craves_satiety_muffin))))
 (:action feast__satiety__cherry__flounder
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (craves_satiety_flounder) (not (craves_satiety_cherry))))
 (:action feast__satiety__cherry__onion
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (craves_satiety_onion) (not (craves_satiety_cherry))))
 (:action feast__satiety__cherry__beef
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (craves_satiety_beef) (not (craves_satiety_cherry))))
 (:action overcome__boils__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_boils_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_boils_tuna)) (fears_boils_entertainment)))
 (:action overcome__dread__curiosity__onion
  :parameters ()
  :precondition (and (craves_dread_onion) (craves_curiosity_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread_onion)) (fears_dread_curiosity)))
 (:action succumb__grief7__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_grief7_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_grief7_entertainment)) (craves_grief7_tuna)))
 (:action succumb__grief7__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_grief7_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_grief7_entertainment)) (craves_grief7_ham)))
 (:action succumb__grief7__entertainment__muffin
  :parameters ()
  :precondition (and (craves_entertainment_muffin) (fears_grief7_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_grief7_entertainment)) (craves_grief7_muffin)))
 (:action overcome__sciatica__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_sciatica_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_sciatica_tuna)) (fears_sciatica_entertainment)))
 (:action overcome__boils__stimulation__tuna
  :parameters ()
  :precondition (and (craves_boils_tuna) (not (v4_4)) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils_tuna)) (fears_boils_stimulation)))
 (:action succumb__jealousy2__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (fears_jealousy2_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_jealousy2_curiosity)) (craves_jealousy2_onion)))
 (:action succumb__jealousy2__curiosity__beef
  :parameters ()
  :precondition (and (craves_curiosity_beef) (fears_jealousy2_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_jealousy2_curiosity)) (craves_jealousy2_beef)))
 (:action succumb__jealousy2__curiosity__flounder
  :parameters ()
  :precondition (and (fears_jealousy2_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_jealousy2_curiosity)) (craves_jealousy2_flounder)))
 (:action succumb__jealousy2__curiosity__cherry
  :parameters ()
  :precondition (and (craves_curiosity_cherry) (fears_jealousy2_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_jealousy2_curiosity)) (craves_jealousy2_cherry)))
 (:action overcome__sciatica__stimulation__tuna
  :parameters ()
  :precondition (and (craves_sciatica_tuna) (not (v4_4)) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_sciatica_stimulation) (not (craves_sciatica_tuna))))
 (:action overcome__anxiety__curiosity__flounder
  :parameters ()
  :precondition (and (craves_anxiety_flounder) (not (v5_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_anxiety_curiosity) (not (craves_anxiety_flounder))))
 (:action overcome__jealousy2__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_jealousy2_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_jealousy2_satiety) (not (craves_jealousy2_cherry))))
 (:action succumb__prostatitis3__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_prostatitis3_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_prostatitis3_ham) (not (fears_prostatitis3_entertainment))))
 (:action succumb__prostatitis3__entertainment__muffin
  :parameters ()
  :precondition (and (fears_prostatitis3_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_prostatitis3_entertainment)) (craves_prostatitis3_muffin)))
 (:action succumb__prostatitis3__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_prostatitis3_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_prostatitis3_tuna) (not (fears_prostatitis3_entertainment))))
 (:action overcome__jealousy2__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_jealousy2_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_jealousy2_stimulation) (not (craves_jealousy2_cherry))))
 (:action succumb__dread__satiety__onion
  :parameters ()
  :precondition (and (fears_dread_satiety) (craves_satiety_onion) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread_onion) (not (fears_dread_satiety))))
 (:action succumb__dread__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_dread_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread_beef) (not (fears_dread_satiety))))
 (:action succumb__dread__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_dread_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread_muffin) (not (fears_dread_satiety))))
 (:action succumb__dread__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_dread_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread_cherry) (not (fears_dread_satiety))))
 (:action overcome__depression__curiosity__beef
  :parameters ()
  :precondition (and (not (v5_4)) (craves_depression_beef) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_depression_beef)) (fears_depression_curiosity)))
 (:action overcome__grief__curiosity__beef
  :parameters ()
  :precondition (and (craves_grief_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_grief_curiosity) (not (craves_grief_beef))))
 (:action feast__entertainment__tuna__beef
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_entertainment_tuna)) (craves_entertainment_beef)))
 (:action feast__entertainment__tuna__ham
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_entertainment_tuna)) (craves_entertainment_ham)))
 (:action feast__entertainment__tuna__muffin
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_entertainment_tuna)) (craves_entertainment_muffin)))
 (:action feast__entertainment__tuna__flounder
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_entertainment_tuna)) (craves_entertainment_flounder)))
 (:action feast__entertainment__muffin__onion
  :parameters ()
  :precondition (and (craves_entertainment_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (not (craves_entertainment_muffin)) (craves_entertainment_onion)))
 (:action feast__entertainment__muffin__tuna
  :parameters ()
  :precondition (and (craves_entertainment_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_entertainment_tuna) (not (craves_entertainment_muffin))))
 (:action feast__entertainment__muffin__ham
  :parameters ()
  :precondition (and (craves_entertainment_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_entertainment_ham) (not (craves_entertainment_muffin))))
 (:action overcome__depression__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (craves_depression_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression_beef)) (fears_depression_satiety)))
 (:action overcome__loneliness__curiosity__onion
  :parameters ()
  :precondition (and (craves_loneliness_onion) (craves_curiosity_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_onion))))
 (:action feast__stimulation__tuna__beef
  :parameters ()
  :precondition (and (craves_stimulation_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (craves_stimulation_beef) (not (craves_stimulation_tuna))))
 (:action feast__stimulation__tuna__flounder
  :parameters ()
  :precondition (and (craves_stimulation_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_stimulation_tuna)) (craves_stimulation_flounder)))
 (:action feast__stimulation__tuna__ham
  :parameters ()
  :precondition (and (craves_stimulation_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (craves_stimulation_ham) (not (craves_stimulation_tuna))))
 (:action feast__stimulation__tuna__muffin
  :parameters ()
  :precondition (and (craves_stimulation_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_stimulation_tuna)) (craves_stimulation_muffin)))
 (:action succumb__anxiety__stimulation__tuna
  :parameters ()
  :precondition (and (craves_stimulation_tuna) (fears_anxiety_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_anxiety_stimulation)) (craves_anxiety_tuna)))
 (:action succumb__anxiety__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_anxiety_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_anxiety_cherry) (not (fears_anxiety_stimulation))))
 (:action succumb__anxiety__stimulation__flounder
  :parameters ()
  :precondition (and (craves_stimulation_flounder) (fears_anxiety_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_anxiety_flounder) (not (fears_anxiety_stimulation))))
 (:action overcome__grief__satiety__beef
  :parameters ()
  :precondition (and (craves_grief_beef) (craves_satiety_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_grief_satiety) (not (craves_grief_beef))))
 (:action feast__stimulation__cherry__onion
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (not (craves_stimulation_cherry)) (craves_stimulation_onion)))
 (:action feast__stimulation__cherry__beef
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (not (craves_stimulation_cherry)) (craves_stimulation_beef)))
 (:action feast__stimulation__cherry__flounder
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (not (craves_stimulation_cherry)) (craves_stimulation_flounder)))
 (:action overcome__angina__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_angina_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_angina_curiosity) (not (craves_angina_onion))))
 (:action overcome__laceration__stimulation__tuna
  :parameters ()
  :precondition (and (craves_laceration_tuna) (not (v4_4)) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_laceration_tuna)) (fears_laceration_stimulation)))
 (:action succumb__dread8__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_dread8_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_dread8_ham) (not (fears_dread8_entertainment))))
 (:action succumb__dread8__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_dread8_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread8_entertainment)) (craves_dread8_tuna)))
 (:action succumb__dread8__entertainment__muffin
  :parameters ()
  :precondition (and (fears_dread8_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_dread8_muffin) (not (fears_dread8_entertainment))))
 (:action overcome__laceration__entertainment__tuna
  :parameters ()
  :precondition (and (craves_laceration_tuna) (craves_entertainment_tuna) (not (v0_4)) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_laceration_entertainment) (not (craves_laceration_tuna))))
 (:action succumb__jealousy__stimulation__flounder
  :parameters ()
  :precondition (and (craves_stimulation_flounder) (fears_jealousy_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_jealousy_flounder) (not (fears_jealousy_stimulation))))
 (:action succumb__jealousy__stimulation__tuna
  :parameters ()
  :precondition (and (fears_jealousy_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_jealousy_tuna) (not (fears_jealousy_stimulation))))
 (:action succumb__jealousy__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_jealousy_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_jealousy_stimulation)) (craves_jealousy_cherry)))
 (:action overcome__hangover__stimulation__tuna
  :parameters ()
  :precondition (and (craves_hangover_tuna) (not (v4_4)) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_hangover_tuna)) (fears_hangover_stimulation)))
 (:action overcome__hangover__entertainment__tuna
  :parameters ()
  :precondition (and (craves_hangover_tuna) (craves_entertainment_tuna) (not (v0_4)) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_hangover_tuna)) (fears_hangover_entertainment)))
 (:action overcome__jealousy__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_jealousy_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy_flounder)) (fears_jealousy_curiosity)))
 (:action succumb__boils4__entertainment__muffin
  :parameters ()
  :precondition (and (fears_boils4_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils4_muffin) (not (fears_boils4_entertainment))))
 (:action succumb__boils4__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_boils4_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils4_ham) (not (fears_boils4_entertainment))))
 (:action succumb__boils4__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_boils4_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils4_tuna) (not (fears_boils4_entertainment))))
 (:action overcome__depression1__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_depression1_muffin) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression1_muffin)) (fears_depression1_entertainment)))
 (:action overcome__dread8__curiosity__muffin
  :parameters ()
  :precondition (and (craves_dread8_muffin) (not (v5_4)) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread8_muffin)) (fears_dread8_curiosity)))
 (:action succumb__angina__curiosity__beef
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_beef)))
 (:action succumb__angina__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_angina_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_tuna)))
 (:action succumb__angina__curiosity__onion
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_angina_onion) (not (fears_angina_curiosity))))
 (:action succumb__angina__curiosity__cherry
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_cherry)))
 (:action succumb__angina__curiosity__flounder
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_flounder)))
 (:action succumb__angina__curiosity__muffin
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_muffin)))
 (:action overcome__dread8__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_dread8_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread8_tuna)) (fears_dread8_curiosity)))
 (:action overcome__dread8__stimulation__muffin
  :parameters ()
  :precondition (and (craves_dread8_muffin) (not (v4_4)) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_dread8_muffin)) (fears_dread8_stimulation)))
 (:action overcome__dread8__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_dread8_ham) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_dread8_ham)) (fears_dread8_stimulation)))
 (:action overcome__dread8__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_tuna) (craves_dread8_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread8_stimulation) (not (craves_dread8_tuna))))
 (:action succumb__angina__satiety__ham
  :parameters ()
  :precondition (and (fears_angina_satiety) (craves_satiety_ham) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_angina_satiety)) (craves_angina_ham)))
 (:action overcome__prostatitis__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_prostatitis_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis_entertainment) (not (craves_prostatitis_beef))))
 (:action succumb__angina__satiety__tuna
  :parameters ()
  :precondition (and (fears_angina_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_angina_tuna) (not (fears_angina_satiety))))
 (:action succumb__angina__satiety__flounder
  :parameters ()
  :precondition (and (fears_angina_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_angina_satiety)) (craves_angina_flounder)))
 (:action succumb__laceration__entertainment__ham
  :parameters ()
  :precondition (and (fears_laceration_entertainment) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_laceration_entertainment)) (craves_laceration_ham)))
 (:action succumb__laceration__entertainment__onion
  :parameters ()
  :precondition (and (fears_laceration_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_laceration_entertainment)) (craves_laceration_onion)))
 (:action succumb__laceration__entertainment__beef
  :parameters ()
  :precondition (and (fears_laceration_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_laceration_beef) (not (fears_laceration_entertainment))))
 (:action succumb__laceration__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_laceration_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_laceration_flounder) (not (fears_laceration_entertainment))))
 (:action succumb__laceration__entertainment__tuna
  :parameters ()
  :precondition (and (fears_laceration_entertainment) (craves_entertainment_tuna) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_laceration_tuna) (not (fears_laceration_entertainment))))
 (:action succumb__laceration__entertainment__muffin
  :parameters ()
  :precondition (and (fears_laceration_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_laceration_entertainment)) (craves_laceration_muffin)))
 (:action overcome__dread8__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_dread8_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread8_entertainment) (not (craves_dread8_tuna))))
 (:action overcome__dread8__entertainment__muffin
  :parameters ()
  :precondition (and (craves_dread8_muffin) (not (v0_4)) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_dread8_muffin)) (fears_dread8_entertainment)))
 (:action overcome__anger__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_beef) (craves_anger_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_anger_stimulation) (not (craves_anger_beef))))
 (:action succumb__loneliness__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_loneliness_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_tuna)))
 (:action succumb__loneliness__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (fears_loneliness_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_onion)))
 (:action overcome__boils4__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_boils4_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils4_muffin)) (fears_boils4_curiosity)))
 (:action succumb__loneliness__curiosity__cherry
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_loneliness_cherry) (not (fears_loneliness_curiosity))))
 (:action succumb__loneliness__curiosity__flounder
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_loneliness_flounder) (not (fears_loneliness_curiosity))))
 (:action succumb__loneliness__curiosity__muffin
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_muffin)))
 (:action succumb__loneliness__curiosity__beef
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_beef)))
 (:action overcome__boils4__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_boils4_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils4_tuna)) (fears_boils4_curiosity)))
 (:action overcome__boils4__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_boils4_tuna) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils4_tuna)) (fears_boils4_stimulation)))
 (:action overcome__boils4__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_boils4_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils4_muffin)) (fears_boils4_stimulation)))
 (:action overcome__boils4__stimulation__ham
  :parameters ()
  :precondition (and (craves_boils4_ham) (not (v4_4)) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils4_ham)) (fears_boils4_stimulation)))
 (:action overcome__abrasion__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_beef) (craves_abrasion_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_abrasion_entertainment) (not (craves_abrasion_beef))))
 (:action overcome__dread8__satiety__muffin
  :parameters ()
  :precondition (and (craves_dread8_muffin) (craves_satiety_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread8_muffin)) (fears_dread8_satiety)))
 (:action succumb__loneliness__satiety__tuna
  :parameters ()
  :precondition (and (fears_loneliness_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_loneliness_satiety)) (craves_loneliness_tuna)))
 (:action overcome__dread8__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (craves_dread8_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread8_ham)) (fears_dread8_satiety)))
 (:action succumb__loneliness__satiety__flounder
  :parameters ()
  :precondition (and (fears_loneliness_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_loneliness_satiety)) (craves_loneliness_flounder)))
 (:action succumb__loneliness__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_loneliness_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_loneliness_satiety)) (craves_loneliness_ham)))
 (:action overcome__dread8__satiety__tuna
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_tuna) (craves_dread8_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_dread8_satiety) (not (craves_dread8_tuna))))
 (:action overcome__boils4__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_muffin) (craves_boils4_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_boils4_muffin)) (fears_boils4_entertainment)))
 (:action feast__curiosity__tuna__flounder
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_curiosity_tuna)) (craves_curiosity_flounder)))
 (:action feast__curiosity__tuna__beef
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_curiosity_tuna)) (craves_curiosity_beef)))
 (:action feast__curiosity__tuna__ham
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_curiosity_tuna)) (craves_curiosity_ham)))
 (:action feast__curiosity__tuna__muffin
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_curiosity_tuna)) (craves_curiosity_muffin)))
 (:action overcome__boils4__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_boils4_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_boils4_entertainment) (not (craves_boils4_tuna))))
 (:action overcome__prostatitis__stimulation__beef
  :parameters ()
  :precondition (and (craves_prostatitis_beef) (not (v4_4)) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_prostatitis_stimulation) (not (craves_prostatitis_beef))))
 (:action feast__curiosity__muffin__tuna
  :parameters ()
  :precondition (and (craves_curiosity_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_curiosity_tuna) (not (craves_curiosity_muffin))))
 (:action feast__curiosity__muffin__onion
  :parameters ()
  :precondition (and (craves_curiosity_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_curiosity_onion) (not (craves_curiosity_muffin))))
 (:action feast__curiosity__muffin__ham
  :parameters ()
  :precondition (and (craves_curiosity_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_curiosity_ham) (not (craves_curiosity_muffin))))
 (:action succumb__hangover__stimulation__flounder
  :parameters ()
  :precondition (and (fears_hangover_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_hangover_flounder) (not (fears_hangover_stimulation))))
 (:action succumb__hangover__stimulation__tuna
  :parameters ()
  :precondition (and (fears_hangover_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_hangover_tuna) (not (fears_hangover_stimulation))))
 (:action succumb__hangover__stimulation__muffin
  :parameters ()
  :precondition (and (fears_hangover_stimulation) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_hangover_stimulation)) (craves_hangover_muffin)))
 (:action succumb__hangover__stimulation__cherry
  :parameters ()
  :precondition (and (fears_hangover_stimulation) (craves_stimulation_cherry) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_hangover_cherry) (not (fears_hangover_stimulation))))
 (:action succumb__hangover__stimulation__ham
  :parameters ()
  :precondition (and (fears_hangover_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_hangover_ham) (not (fears_hangover_stimulation))))
 (:action succumb__hangover__stimulation__onion
  :parameters ()
  :precondition (and (fears_hangover_stimulation) (craves_stimulation_onion) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_hangover_stimulation)) (craves_hangover_onion)))
 (:action succumb__hangover__stimulation__beef
  :parameters ()
  :precondition (and (fears_hangover_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_hangover_stimulation)) (craves_hangover_beef)))
 (:action overcome__dread__entertainment__onion
  :parameters ()
  :precondition (and (craves_dread_onion) (not (v0_4)) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread_entertainment) (not (craves_dread_onion))))
 (:action overcome__dread__entertainment__muffin
  :parameters ()
  :precondition (and (craves_dread_muffin) (not (v0_4)) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread_entertainment) (not (craves_dread_muffin))))
 (:action overcome__dread__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_dread_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread_entertainment) (not (craves_dread_beef))))
 (:action overcome__boils4__satiety__tuna
  :parameters ()
  :precondition (and (craves_boils4_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_boils4_satiety) (not (craves_boils4_tuna))))
 (:action feast__satiety__flounder__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of)) (not (craves_satiety_flounder)) (craves_satiety_tuna)))
 (:action overcome__boils4__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v3_4)) (craves_boils4_muffin) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_boils4_muffin)) (fears_boils4_satiety)))
 (:action overcome__boils4__satiety__ham
  :parameters ()
  :precondition (and (craves_boils4_ham) (craves_satiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_boils4_ham)) (fears_boils4_satiety)))
 (:action feast__satiety__flounder__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of)) (not (craves_satiety_flounder)) (craves_satiety_cherry)))
 (:action feast__satiety__tuna__beef
  :parameters ()
  :precondition (and (craves_satiety_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (craves_satiety_beef) (not (craves_satiety_tuna))))
 (:action feast__satiety__tuna__muffin
  :parameters ()
  :precondition (and (craves_satiety_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_satiety_tuna)) (craves_satiety_muffin)))
 (:action feast__satiety__tuna__flounder
  :parameters ()
  :precondition (and (craves_satiety_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (craves_satiety_flounder) (not (craves_satiety_tuna))))
 (:action feast__satiety__tuna__ham
  :parameters ()
  :precondition (and (craves_satiety_tuna) (not (v7_4)) (not (of)))
  :effect (and (when (z_7_0) (v7_0)) (when (not (z_7_0)) (not (v7_0))) (when (z_7_1) (v7_1)) (when (not (z_7_1)) (not (v7_1))) (when (z_7_2) (v7_2)) (when (not (z_7_2)) (not (v7_2))) (when (z_7_3) (v7_3)) (when (not (z_7_3)) (not (v7_3))) (when (z_7_4) (v7_4)) (when (not (z_7_4)) (not (v7_4))) (when (and (v7_4) (not (z_7_4))) (of)) (not (craves_satiety_tuna)) (craves_satiety_ham)))
 (:action succumb__jealousy__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_jealousy_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_jealousy_tuna) (not (fears_jealousy_curiosity))))
 (:action succumb__jealousy__curiosity__onion
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_jealousy_onion) (not (fears_jealousy_curiosity))))
 (:action succumb__jealousy__curiosity__cherry
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_jealousy_curiosity)) (craves_jealousy_cherry)))
 (:action succumb__jealousy__curiosity__flounder
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_jealousy_flounder) (not (fears_jealousy_curiosity))))
 (:action succumb__jealousy__curiosity__muffin
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_jealousy_muffin) (not (fears_jealousy_curiosity))))
 (:action overcome__abrasion__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_beef) (craves_abrasion_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_abrasion_stimulation) (not (craves_abrasion_beef))))
 (:action overcome__dread__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (craves_dread_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_dread_satiety) (not (craves_dread_beef))))
 (:action overcome__dread__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v3_4)) (craves_dread_cherry) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread_cherry)) (fears_dread_satiety)))
 (:action overcome__dread__satiety__muffin
  :parameters ()
  :precondition (and (craves_dread_muffin) (craves_satiety_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread_muffin)) (fears_dread_satiety)))
 (:action succumb__laceration__stimulation__muffin
  :parameters ()
  :precondition (and (fears_laceration_stimulation) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_laceration_stimulation)) (craves_laceration_muffin)))
 (:action succumb__laceration__stimulation__cherry
  :parameters ()
  :precondition (and (fears_laceration_stimulation) (craves_stimulation_cherry) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_laceration_stimulation)) (craves_laceration_cherry)))
 (:action succumb__laceration__stimulation__ham
  :parameters ()
  :precondition (and (fears_laceration_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_laceration_stimulation)) (craves_laceration_ham)))
 (:action succumb__laceration__stimulation__onion
  :parameters ()
  :precondition (and (fears_laceration_stimulation) (craves_stimulation_onion) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_laceration_stimulation)) (craves_laceration_onion)))
 (:action succumb__laceration__stimulation__beef
  :parameters ()
  :precondition (and (fears_laceration_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_laceration_beef) (not (fears_laceration_stimulation))))
 (:action succumb__laceration__stimulation__flounder
  :parameters ()
  :precondition (and (fears_laceration_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_laceration_flounder) (not (fears_laceration_stimulation))))
 (:action succumb__laceration__stimulation__tuna
  :parameters ()
  :precondition (and (fears_laceration_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_laceration_tuna) (not (fears_laceration_stimulation))))
 (:action overcome__dread__curiosity__beef
  :parameters ()
  :precondition (and (craves_dread_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_dread_curiosity) (not (craves_dread_beef))))
 (:action overcome__dread__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_cherry) (craves_dread_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread_cherry)) (fears_dread_curiosity)))
 (:action overcome__dread__curiosity__muffin
  :parameters ()
  :precondition (and (craves_dread_muffin) (not (v5_4)) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread_muffin)) (fears_dread_curiosity)))
 (:action succumb__hangover__entertainment__muffin
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_hangover_entertainment)) (craves_hangover_muffin)))
 (:action succumb__hangover__entertainment__ham
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_hangover_ham) (not (fears_hangover_entertainment))))
 (:action feast__satiety__ham__muffin
  :parameters ()
  :precondition (and (craves_satiety_ham) (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (craves_satiety_muffin) (not (craves_satiety_ham))))
 (:action succumb__hangover__entertainment__onion
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_hangover_onion) (not (fears_hangover_entertainment))))
 (:action succumb__hangover__entertainment__beef
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_hangover_entertainment)) (craves_hangover_beef)))
 (:action succumb__hangover__entertainment__flounder
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_flounder) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_hangover_flounder) (not (fears_hangover_entertainment))))
 (:action feast__satiety__ham__tuna
  :parameters ()
  :precondition (and (craves_satiety_ham) (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (craves_satiety_tuna) (not (craves_satiety_ham))))
 (:action succumb__hangover__entertainment__tuna
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_tuna) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_hangover_tuna) (not (fears_hangover_entertainment))))
 (:action overcome__dread__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_dread_beef) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread_stimulation) (not (craves_dread_beef))))
 (:action overcome__dread__stimulation__onion
  :parameters ()
  :precondition (and (craves_dread_onion) (craves_stimulation_onion) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread_stimulation) (not (craves_dread_onion))))
 (:action overcome__dread__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_dread_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread_stimulation) (not (craves_dread_cherry))))
 (:action overcome__dread__stimulation__muffin
  :parameters ()
  :precondition (and (craves_dread_muffin) (not (v4_4)) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread_stimulation) (not (craves_dread_muffin))))
 (:action overcome__boils__satiety__tuna
  :parameters ()
  :precondition (and (craves_boils_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_boils_tuna)) (fears_boils_satiety)))
 (:action succumb__grief7__entertainment__beef
  :parameters ()
  :precondition (and (craves_entertainment_beef) (fears_grief7_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_grief7_entertainment)) (craves_grief7_beef)))
 (:action succumb__grief7__entertainment__onion
  :parameters ()
  :precondition (and (craves_entertainment_onion) (fears_grief7_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_grief7_onion) (not (fears_grief7_entertainment))))
 (:action succumb__grief7__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_grief7_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_grief7_entertainment)) (craves_grief7_flounder)))
 (:action overcome__boils__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (craves_boils_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils_tuna)) (fears_boils_curiosity)))
 (:action overcome__sciatica__satiety__tuna
  :parameters ()
  :precondition (and (craves_sciatica_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_sciatica_tuna)) (fears_sciatica_satiety)))
 (:action overcome__anxiety__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_anxiety_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anxiety_entertainment) (not (craves_anxiety_flounder))))
 (:action overcome__anxiety__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_anxiety_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_anxiety_tuna)) (fears_anxiety_entertainment)))
 (:action succumb__jealousy2__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_jealousy2_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_jealousy2_curiosity)) (craves_jealousy2_tuna)))
 (:action succumb__jealousy2__curiosity__muffin
  :parameters ()
  :precondition (and (craves_curiosity_muffin) (fears_jealousy2_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_jealousy2_curiosity)) (craves_jealousy2_muffin)))
 (:action overcome__sciatica__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (craves_sciatica_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_tuna))))
 (:action succumb__jealousy2__stimulation__beef
  :parameters ()
  :precondition (and (craves_stimulation_beef) (fears_jealousy2_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_jealousy2_beef) (not (fears_jealousy2_stimulation))))
 (:action succumb__jealousy2__stimulation__tuna
  :parameters ()
  :precondition (and (fears_jealousy2_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_jealousy2_tuna) (not (fears_jealousy2_stimulation))))
 (:action succumb__jealousy2__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_jealousy2_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_jealousy2_onion) (not (fears_jealousy2_stimulation))))
 (:action succumb__jealousy2__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_jealousy2_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_jealousy2_stimulation)) (craves_jealousy2_cherry)))
 (:action succumb__jealousy2__stimulation__flounder
  :parameters ()
  :precondition (and (craves_stimulation_flounder) (fears_jealousy2_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_jealousy2_stimulation)) (craves_jealousy2_flounder)))
 (:action succumb__jealousy2__stimulation__ham
  :parameters ()
  :precondition (and (craves_stimulation_ham) (fears_jealousy2_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_jealousy2_stimulation)) (craves_jealousy2_ham)))
 (:action succumb__jealousy2__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_jealousy2_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_jealousy2_stimulation)) (craves_jealousy2_muffin)))
 (:action succumb__jealousy2__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_jealousy2_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_jealousy2_satiety)) (craves_jealousy2_ham)))
 (:action succumb__jealousy2__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_jealousy2_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_jealousy2_onion) (not (fears_jealousy2_satiety))))
 (:action succumb__jealousy2__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_jealousy2_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_jealousy2_beef) (not (fears_jealousy2_satiety))))
 (:action succumb__jealousy2__satiety__flounder
  :parameters ()
  :precondition (and (craves_satiety_flounder) (fears_jealousy2_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_jealousy2_flounder) (not (fears_jealousy2_satiety))))
 (:action succumb__jealousy2__satiety__tuna
  :parameters ()
  :precondition (and (craves_satiety_tuna) (fears_jealousy2_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_jealousy2_tuna) (not (fears_jealousy2_satiety))))
 (:action succumb__jealousy2__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_jealousy2_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_jealousy2_satiety)) (craves_jealousy2_muffin)))
 (:action succumb__jealousy2__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_jealousy2_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_jealousy2_cherry) (not (fears_jealousy2_satiety))))
 (:action succumb__grief__curiosity__cherry
  :parameters ()
  :precondition (and (fears_grief_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_grief_curiosity)) (craves_grief_cherry)))
 (:action succumb__grief__curiosity__flounder
  :parameters ()
  :precondition (and (fears_grief_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_grief_curiosity)) (craves_grief_flounder)))
 (:action succumb__grief__curiosity__muffin
  :parameters ()
  :precondition (and (fears_grief_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_grief_curiosity)) (craves_grief_muffin)))
 (:action overcome__anxiety__satiety__tuna
  :parameters ()
  :precondition (and (craves_anxiety_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_anxiety_tuna)) (fears_anxiety_satiety)))
 (:action overcome__anxiety__satiety__flounder
  :parameters ()
  :precondition (and (craves_anxiety_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_anxiety_flounder)) (fears_anxiety_satiety)))
 (:action overcome__anxiety__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_anxiety_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_anxiety_cherry)) (fears_anxiety_satiety)))
 (:action succumb__grief__curiosity__beef
  :parameters ()
  :precondition (and (fears_grief_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_grief_curiosity)) (craves_grief_beef)))
 (:action succumb__grief__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_grief_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_grief_curiosity)) (craves_grief_tuna)))
 (:action succumb__grief__curiosity__onion
  :parameters ()
  :precondition (and (fears_grief_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_grief_curiosity)) (craves_grief_onion)))
 (:action overcome__jealousy2__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_jealousy2_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy2_entertainment) (not (craves_jealousy2_onion))))
 (:action overcome__jealousy2__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_jealousy2_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy2_entertainment) (not (craves_jealousy2_flounder))))
 (:action overcome__jealousy2__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_beef) (craves_jealousy2_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy2_entertainment) (not (craves_jealousy2_beef))))
 (:action succumb__grief__satiety__tuna
  :parameters ()
  :precondition (and (fears_grief_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_grief_tuna) (not (fears_grief_satiety))))
 (:action succumb__grief__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_grief_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_grief_satiety)) (craves_grief_onion)))
 (:action succumb__grief__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_grief_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_grief_cherry) (not (fears_grief_satiety))))
 (:action succumb__grief__satiety__flounder
  :parameters ()
  :precondition (and (fears_grief_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_grief_flounder) (not (fears_grief_satiety))))
 (:action succumb__grief__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_grief_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_grief_ham) (not (fears_grief_satiety))))
 (:action succumb__grief__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_grief_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_grief_satiety)) (craves_grief_muffin)))
 (:action overcome__anxiety__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (craves_anxiety_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_anxiety_tuna)) (fears_anxiety_curiosity)))
 (:action overcome__anxiety__curiosity__cherry
  :parameters ()
  :precondition (and (craves_anxiety_cherry) (not (v5_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_anxiety_cherry)) (fears_anxiety_curiosity)))
 (:action succumb__grief__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_grief_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_grief_satiety)) (craves_grief_beef)))
 (:action overcome__anxiety__stimulation__tuna
  :parameters ()
  :precondition (and (craves_anxiety_tuna) (not (v4_4)) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_anxiety_tuna)) (fears_anxiety_stimulation)))
 (:action overcome__anxiety__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_anxiety_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_anxiety_cherry)) (fears_anxiety_stimulation)))
 (:action succumb__depression__curiosity__muffin
  :parameters ()
  :precondition (and (craves_curiosity_muffin) (fears_depression_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression_muffin) (not (fears_depression_curiosity))))
 (:action overcome__jealousy2__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (not (v3_4)) (craves_jealousy2_beef) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_jealousy2_satiety) (not (craves_jealousy2_beef))))
 (:action succumb__depression__curiosity__beef
  :parameters ()
  :precondition (and (craves_curiosity_beef) (fears_depression_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression_beef) (not (fears_depression_curiosity))))
 (:action overcome__jealousy2__satiety__onion
  :parameters ()
  :precondition (and (craves_jealousy2_onion) (craves_satiety_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_jealousy2_onion)) (fears_jealousy2_satiety)))
 (:action succumb__depression__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_depression_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression_tuna) (not (fears_depression_curiosity))))
 (:action overcome__jealousy2__satiety__flounder
  :parameters ()
  :precondition (and (craves_jealousy2_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_jealousy2_satiety) (not (craves_jealousy2_flounder))))
 (:action succumb__depression__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (fears_depression_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression_onion) (not (fears_depression_curiosity))))
 (:action succumb__depression__curiosity__cherry
  :parameters ()
  :precondition (and (fears_depression_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression_cherry) (not (fears_depression_curiosity))))
 (:action succumb__depression__curiosity__flounder
  :parameters ()
  :precondition (and (fears_depression_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression_flounder) (not (fears_depression_curiosity))))
 (:action succumb__prostatitis3__entertainment__beef
  :parameters ()
  :precondition (and (fears_prostatitis3_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_prostatitis3_entertainment)) (craves_prostatitis3_beef)))
 (:action succumb__prostatitis3__entertainment__onion
  :parameters ()
  :precondition (and (fears_prostatitis3_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_prostatitis3_onion) (not (fears_prostatitis3_entertainment))))
 (:action succumb__prostatitis3__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_prostatitis3_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_prostatitis3_flounder) (not (fears_prostatitis3_entertainment))))
 (:action overcome__grief7__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_grief7_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief7_entertainment) (not (craves_grief7_tuna))))
 (:action overcome__grief7__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_grief7_muffin) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief7_entertainment) (not (craves_grief7_muffin))))
 (:action succumb__depression__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_depression_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_depression_satiety)) (craves_depression_cherry)))
 (:action succumb__depression__satiety__flounder
  :parameters ()
  :precondition (and (fears_depression_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_depression_satiety)) (craves_depression_flounder)))
 (:action succumb__depression__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_depression_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_depression_ham) (not (fears_depression_satiety))))
 (:action succumb__depression__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_depression_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_depression_muffin) (not (fears_depression_satiety))))
 (:action overcome__jealousy2__curiosity__beef
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_beef) (craves_jealousy2_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy2_beef)) (fears_jealousy2_curiosity)))
 (:action overcome__jealousy2__curiosity__onion
  :parameters ()
  :precondition (and (craves_jealousy2_onion) (craves_curiosity_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy2_onion)) (fears_jealousy2_curiosity)))
 (:action overcome__jealousy2__curiosity__flounder
  :parameters ()
  :precondition (and (craves_jealousy2_flounder) (not (v5_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy2_flounder)) (fears_jealousy2_curiosity)))
 (:action succumb__depression__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_depression_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_depression_satiety)) (craves_depression_beef)))
 (:action succumb__depression__satiety__tuna
  :parameters ()
  :precondition (and (fears_depression_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_depression_satiety)) (craves_depression_tuna)))
 (:action succumb__depression__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_depression_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_depression_satiety)) (craves_depression_onion)))
 (:action overcome__jealousy2__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_beef) (craves_jealousy2_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_jealousy2_stimulation) (not (craves_jealousy2_beef))))
 (:action overcome__jealousy2__stimulation__onion
  :parameters ()
  :precondition (and (craves_jealousy2_onion) (craves_stimulation_onion) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_jealousy2_onion)) (fears_jealousy2_stimulation)))
 (:action overcome__jealousy2__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_jealousy2_flounder) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_jealousy2_stimulation) (not (craves_jealousy2_flounder))))
 (:action overcome__grief7__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (craves_grief7_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_grief7_muffin)) (fears_grief7_satiety)))
 (:action overcome__grief7__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (craves_grief7_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_grief7_satiety) (not (craves_grief7_ham))))
 (:action succumb__dread__satiety__flounder
  :parameters ()
  :precondition (and (fears_dread_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_dread_satiety)) (craves_dread_flounder)))
 (:action succumb__dread__satiety__tuna
  :parameters ()
  :precondition (and (fears_dread_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread_tuna) (not (fears_dread_satiety))))
 (:action overcome__grief7__satiety__tuna
  :parameters ()
  :precondition (and (not (v3_4)) (craves_grief7_tuna) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_grief7_tuna)) (fears_grief7_satiety)))
 (:action overcome__prostatitis3__entertainment__muffin
  :parameters ()
  :precondition (and (craves_prostatitis3_muffin) (not (v0_4)) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_prostatitis3_muffin)) (fears_prostatitis3_entertainment)))
 (:action succumb__boils__entertainment__beef
  :parameters ()
  :precondition (and (fears_boils_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_boils_entertainment)) (craves_boils_beef)))
 (:action succumb__boils__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_boils_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils_tuna) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__onion
  :parameters ()
  :precondition (and (fears_boils_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils_onion) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_boils_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils_flounder) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__ham
  :parameters ()
  :precondition (and (fears_boils_entertainment) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils_ham) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__muffin
  :parameters ()
  :precondition (and (fears_boils_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_boils_entertainment)) (craves_boils_muffin)))
 (:action overcome__prostatitis3__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_prostatitis3_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_prostatitis3_tuna)) (fears_prostatitis3_entertainment)))
 (:action overcome__depression__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_beef) (craves_depression_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_depression_stimulation) (not (craves_depression_beef))))
 (:action feast__entertainment__onion__beef
  :parameters ()
  :precondition (and (craves_entertainment_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (craves_entertainment_beef) (not (craves_entertainment_onion))))
 (:action feast__entertainment__onion__muffin
  :parameters ()
  :precondition (and (craves_entertainment_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (craves_entertainment_muffin) (not (craves_entertainment_onion))))
 (:action overcome__grief7__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_grief7_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_grief7_tuna)) (fears_grief7_curiosity)))
 (:action feast__entertainment__onion__cherry
  :parameters ()
  :precondition (and (craves_entertainment_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (craves_entertainment_cherry) (not (craves_entertainment_onion))))
 (:action overcome__grief7__curiosity__muffin
  :parameters ()
  :precondition (and (craves_grief7_muffin) (not (v5_4)) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_grief7_muffin)) (fears_grief7_curiosity)))
 (:action succumb__boils__stimulation__ham
  :parameters ()
  :precondition (and (fears_boils_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_boils_stimulation)) (craves_boils_ham)))
 (:action succumb__boils__stimulation__muffin
  :parameters ()
  :precondition (and (fears_boils_stimulation) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_boils_stimulation)) (craves_boils_muffin)))
 (:action overcome__grief7__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_grief7_muffin) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief7_stimulation) (not (craves_grief7_muffin))))
 (:action overcome__grief7__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_grief7_ham) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief7_stimulation) (not (craves_grief7_ham))))
 (:action succumb__boils__stimulation__beef
  :parameters ()
  :precondition (and (fears_boils_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_boils_stimulation)) (craves_boils_beef)))
 (:action succumb__boils__stimulation__tuna
  :parameters ()
  :precondition (and (fears_boils_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_boils_stimulation)) (craves_boils_tuna)))
 (:action succumb__boils__stimulation__onion
  :parameters ()
  :precondition (and (fears_boils_stimulation) (craves_stimulation_onion) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_boils_onion) (not (fears_boils_stimulation))))
 (:action succumb__boils__stimulation__cherry
  :parameters ()
  :precondition (and (fears_boils_stimulation) (craves_stimulation_cherry) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_boils_stimulation)) (craves_boils_cherry)))
 (:action succumb__boils__stimulation__flounder
  :parameters ()
  :precondition (and (fears_boils_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_boils_stimulation)) (craves_boils_flounder)))
 (:action feast__entertainment__beef__cherry
  :parameters ()
  :precondition (and (craves_entertainment_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (not (craves_entertainment_beef)) (craves_entertainment_cherry)))
 (:action overcome__grief7__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_grief7_tuna) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_grief7_tuna)) (fears_grief7_stimulation)))
 (:action feast__entertainment__beef__tuna
  :parameters ()
  :precondition (and (craves_entertainment_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (craves_entertainment_tuna) (not (craves_entertainment_beef))))
 (:action feast__entertainment__beef__onion
  :parameters ()
  :precondition (and (craves_entertainment_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (not (craves_entertainment_beef)) (craves_entertainment_onion)))
 (:action overcome__depression__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_depression_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression_beef)) (fears_depression_entertainment)))
 (:action overcome__prostatitis3__satiety__tuna
  :parameters ()
  :precondition (and (craves_prostatitis3_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_prostatitis3_tuna)) (fears_prostatitis3_satiety)))
 (:action succumb__abrasion__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_abrasion_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_abrasion_satiety)) (craves_abrasion_onion)))
 (:action succumb__abrasion__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_abrasion_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_abrasion_satiety)) (craves_abrasion_beef)))
 (:action succumb__abrasion__satiety__flounder
  :parameters ()
  :precondition (and (fears_abrasion_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_abrasion_satiety)) (craves_abrasion_flounder)))
 (:action overcome__prostatitis3__satiety__muffin
  :parameters ()
  :precondition (and (craves_prostatitis3_muffin) (craves_satiety_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_prostatitis3_muffin)) (fears_prostatitis3_satiety)))
 (:action succumb__abrasion__satiety__tuna
  :parameters ()
  :precondition (and (fears_abrasion_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_abrasion_satiety)) (craves_abrasion_tuna)))
 (:action overcome__prostatitis3__satiety__ham
  :parameters ()
  :precondition (and (craves_prostatitis3_ham) (craves_satiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_prostatitis3_ham)) (fears_prostatitis3_satiety)))
 (:action succumb__abrasion__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_abrasion_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_abrasion_satiety)) (craves_abrasion_muffin)))
 (:action succumb__abrasion__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_abrasion_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_abrasion_satiety)) (craves_abrasion_cherry)))
 (:action succumb__abrasion__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_abrasion_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_abrasion_satiety)) (craves_abrasion_ham)))
 (:action feast__entertainment__flounder__tuna
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v8_4)) (not (of)))
  :effect (and (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of)) (craves_entertainment_tuna) (not (craves_entertainment_flounder))))
 (:action feast__entertainment__flounder__cherry
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v8_4)) (not (of)))
  :effect (and (when (z_8_0) (v8_0)) (when (not (z_8_0)) (not (v8_0))) (when (z_8_1) (v8_1)) (when (not (z_8_1)) (not (v8_1))) (when (z_8_2) (v8_2)) (when (not (z_8_2)) (not (v8_2))) (when (z_8_3) (v8_3)) (when (not (z_8_3)) (not (v8_3))) (when (z_8_4) (v8_4)) (when (not (z_8_4)) (not (v8_4))) (when (and (v8_4) (not (z_8_4))) (of)) (not (craves_entertainment_flounder)) (craves_entertainment_cherry)))
 (:action overcome__grief__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_grief_beef) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief_stimulation) (not (craves_grief_beef))))
 (:action succumb__dread__curiosity__onion
  :parameters ()
  :precondition (and (fears_dread_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_dread_onion) (not (fears_dread_curiosity))))
 (:action overcome__prostatitis3__curiosity__muffin
  :parameters ()
  :precondition (and (craves_prostatitis3_muffin) (not (v5_4)) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_prostatitis3_muffin)) (fears_prostatitis3_curiosity)))
 (:action succumb__dread__curiosity__beef
  :parameters ()
  :precondition (and (fears_dread_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_dread_curiosity)) (craves_dread_beef)))
 (:action succumb__dread__curiosity__flounder
  :parameters ()
  :precondition (and (fears_dread_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_dread_curiosity)) (craves_dread_flounder)))
 (:action succumb__dread__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_dread_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_dread_tuna) (not (fears_dread_curiosity))))
 (:action succumb__dread__curiosity__muffin
  :parameters ()
  :precondition (and (fears_dread_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_dread_muffin) (not (fears_dread_curiosity))))
 (:action succumb__dread__curiosity__cherry
  :parameters ()
  :precondition (and (fears_dread_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_dread_cherry) (not (fears_dread_curiosity))))
 (:action feast__stimulation__onion__beef
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (craves_stimulation_beef) (not (craves_stimulation_onion))))
 (:action feast__stimulation__onion__muffin
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (craves_stimulation_muffin) (not (craves_stimulation_onion))))
 (:action overcome__prostatitis3__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_prostatitis3_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_prostatitis3_tuna)) (fears_prostatitis3_curiosity)))
 (:action feast__stimulation__onion__cherry
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v1_4)) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (v1_4) (not (z_17_4))) (of)) (craves_stimulation_cherry) (not (craves_stimulation_onion))))
 (:action overcome__prostatitis3__stimulation__muffin
  :parameters ()
  :precondition (and (craves_prostatitis3_muffin) (not (v4_4)) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_prostatitis3_muffin)) (fears_prostatitis3_stimulation)))
 (:action overcome__prostatitis3__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_prostatitis3_ham) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_prostatitis3_ham)) (fears_prostatitis3_stimulation)))
 (:action feast__stimulation__beef__tuna
  :parameters ()
  :precondition (and (craves_stimulation_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (craves_stimulation_tuna) (not (craves_stimulation_beef))))
 (:action feast__stimulation__beef__onion
  :parameters ()
  :precondition (and (craves_stimulation_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (not (craves_stimulation_beef)) (craves_stimulation_onion)))
 (:action feast__stimulation__beef__cherry
  :parameters ()
  :precondition (and (craves_stimulation_beef) (not (v6_4)) (not (of)))
  :effect (and (when (z_5_0) (v6_0)) (when (not (z_5_0)) (not (v6_0))) (when (z_5_1) (v6_1)) (when (not (z_5_1)) (not (v6_1))) (when (z_5_2) (v6_2)) (when (not (z_5_2)) (not (v6_2))) (when (z_5_3) (v6_3)) (when (not (z_5_3)) (not (v6_3))) (when (z_5_4) (v6_4)) (when (not (z_5_4)) (not (v6_4))) (when (and (v6_4) (not (z_5_4))) (of)) (craves_stimulation_cherry) (not (craves_stimulation_beef))))
 (:action overcome__prostatitis3__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_prostatitis3_tuna) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_prostatitis3_tuna)) (fears_prostatitis3_stimulation)))
 (:action succumb__dread__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_dread_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread_ham) (not (fears_dread_satiety))))
 (:action overcome__grief__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_grief_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief_entertainment) (not (craves_grief_beef))))
 (:action overcome__loneliness__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_cherry) (craves_loneliness_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_loneliness_cherry)) (fears_loneliness_curiosity)))
 (:action overcome__loneliness__curiosity__muffin
  :parameters ()
  :precondition (and (craves_loneliness_muffin) (not (v5_4)) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_muffin))))
 (:action overcome__loneliness__curiosity__beef
  :parameters ()
  :precondition (and (not (v5_4)) (craves_loneliness_beef) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_beef))))
 (:action overcome__loneliness__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_beef) (craves_loneliness_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_loneliness_stimulation) (not (craves_loneliness_beef))))
 (:action overcome__loneliness__stimulation__onion
  :parameters ()
  :precondition (and (craves_loneliness_onion) (craves_stimulation_onion) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_loneliness_stimulation) (not (craves_loneliness_onion))))
 (:action overcome__loneliness__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_loneliness_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_loneliness_cherry)) (fears_loneliness_stimulation)))
 (:action overcome__loneliness__stimulation__muffin
  :parameters ()
  :precondition (and (craves_loneliness_muffin) (not (v4_4)) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_loneliness_stimulation) (not (craves_loneliness_muffin))))
 (:action succumb__anxiety__curiosity__flounder
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_anxiety_flounder) (not (fears_anxiety_curiosity))))
 (:action succumb__anxiety__curiosity__tuna
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_tuna) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_anxiety_curiosity)) (craves_anxiety_tuna)))
 (:action succumb__anxiety__curiosity__muffin
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_anxiety_muffin) (not (fears_anxiety_curiosity))))
 (:action succumb__anxiety__curiosity__cherry
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_anxiety_cherry) (not (fears_anxiety_curiosity))))
 (:action feast__stimulation__muffin__tuna
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_stimulation_tuna) (not (craves_stimulation_muffin))))
 (:action feast__stimulation__muffin__onion
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_stimulation_onion) (not (craves_stimulation_muffin))))
 (:action succumb__anxiety__curiosity__onion
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_anxiety_curiosity)) (craves_anxiety_onion)))
 (:action succumb__anxiety__curiosity__beef
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_anxiety_curiosity)) (craves_anxiety_beef)))
 (:action feast__stimulation__muffin__ham
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (not (v10_4)) (not (of)))
  :effect (and (when (z_10_0) (v10_0)) (when (not (z_10_0)) (not (v10_0))) (when (z_10_1) (v10_1)) (when (not (z_10_1)) (not (v10_1))) (when (z_10_2) (v10_2)) (when (not (z_10_2)) (not (v10_2))) (when (z_10_3) (v10_3)) (when (not (z_10_3)) (not (v10_3))) (when (z_10_4) (v10_4)) (when (not (z_10_4)) (not (v10_4))) (when (and (v10_4) (not (z_10_4))) (of)) (craves_stimulation_ham) (not (craves_stimulation_muffin))))
 (:action succumb__anxiety__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_anxiety_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_anxiety_onion) (not (fears_anxiety_stimulation))))
 (:action succumb__anxiety__stimulation__ham
  :parameters ()
  :precondition (and (craves_stimulation_ham) (fears_anxiety_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_anxiety_ham) (not (fears_anxiety_stimulation))))
 (:action succumb__anxiety__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_anxiety_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_anxiety_muffin) (not (fears_anxiety_stimulation))))
 (:action succumb__anxiety__stimulation__beef
  :parameters ()
  :precondition (and (craves_stimulation_beef) (fears_anxiety_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_anxiety_beef) (not (fears_anxiety_stimulation))))
 (:action overcome__loneliness__entertainment__muffin
  :parameters ()
  :precondition (and (craves_loneliness_muffin) (not (v0_4)) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_loneliness_muffin)) (fears_loneliness_entertainment)))
 (:action overcome__loneliness__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_loneliness_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_loneliness_entertainment) (not (craves_loneliness_beef))))
 (:action overcome__loneliness__entertainment__onion
  :parameters ()
  :precondition (and (craves_loneliness_onion) (not (v0_4)) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_loneliness_entertainment) (not (craves_loneliness_onion))))
 (:action overcome__angina__curiosity__beef
  :parameters ()
  :precondition (and (craves_angina_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_angina_curiosity) (not (craves_angina_beef))))
 (:action overcome__angina__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_angina_cherry) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_angina_curiosity) (not (craves_angina_cherry))))
 (:action overcome__angina__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_angina_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_angina_curiosity) (not (craves_angina_muffin))))
 (:action succumb__sciatica__entertainment__ham
  :parameters ()
  :precondition (and (fears_sciatica_entertainment) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_sciatica_ham) (not (fears_sciatica_entertainment))))
 (:action succumb__sciatica__entertainment__muffin
  :parameters ()
  :precondition (and (fears_sciatica_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_sciatica_entertainment)) (craves_sciatica_muffin)))
 (:action succumb__sciatica__entertainment__beef
  :parameters ()
  :precondition (and (fears_sciatica_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_sciatica_beef) (not (fears_sciatica_entertainment))))
 (:action succumb__sciatica__entertainment__tuna
  :parameters ()
  :precondition (and (fears_sciatica_entertainment) (craves_entertainment_tuna) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_sciatica_tuna) (not (fears_sciatica_entertainment))))
 (:action succumb__sciatica__entertainment__onion
  :parameters ()
  :precondition (and (fears_sciatica_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_sciatica_onion) (not (fears_sciatica_entertainment))))
 (:action feast__stimulation__ham__tuna
  :parameters ()
  :precondition (and (craves_stimulation_ham) (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (craves_stimulation_tuna) (not (craves_stimulation_ham))))
 (:action succumb__sciatica__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_sciatica_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_sciatica_entertainment)) (craves_sciatica_flounder)))
 (:action feast__stimulation__ham__muffin
  :parameters ()
  :precondition (and (craves_stimulation_ham) (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (not (craves_stimulation_ham)) (craves_stimulation_muffin)))
 (:action overcome__angina__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_angina_beef) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_angina_stimulation) (not (craves_angina_beef))))
 (:action overcome__angina__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_angina_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_angina_stimulation) (not (craves_angina_onion))))
 (:action overcome__angina__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_angina_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_angina_stimulation) (not (craves_angina_cherry))))
 (:action overcome__angina__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_angina_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_angina_stimulation) (not (craves_angina_muffin))))
 (:action overcome__loneliness__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (craves_loneliness_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_loneliness_beef)) (fears_loneliness_satiety)))
 (:action overcome__loneliness__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v3_4)) (craves_loneliness_cherry) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_loneliness_cherry)) (fears_loneliness_satiety)))
 (:action overcome__loneliness__satiety__muffin
  :parameters ()
  :precondition (and (craves_loneliness_muffin) (craves_satiety_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_loneliness_muffin)) (fears_loneliness_satiety)))
 (:action succumb__sciatica__stimulation__cherry
  :parameters ()
  :precondition (and (fears_sciatica_stimulation) (craves_stimulation_cherry) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_sciatica_cherry) (not (fears_sciatica_stimulation))))
 (:action succumb__sciatica__stimulation__flounder
  :parameters ()
  :precondition (and (fears_sciatica_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_sciatica_stimulation)) (craves_sciatica_flounder)))
 (:action succumb__sciatica__stimulation__ham
  :parameters ()
  :precondition (and (fears_sciatica_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_sciatica_stimulation)) (craves_sciatica_ham)))
 (:action succumb__sciatica__stimulation__muffin
  :parameters ()
  :precondition (and (fears_sciatica_stimulation) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_sciatica_stimulation)) (craves_sciatica_muffin)))
 (:action succumb__sciatica__stimulation__beef
  :parameters ()
  :precondition (and (fears_sciatica_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_sciatica_beef) (not (fears_sciatica_stimulation))))
 (:action succumb__sciatica__stimulation__tuna
  :parameters ()
  :precondition (and (fears_sciatica_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_sciatica_stimulation)) (craves_sciatica_tuna)))
 (:action succumb__sciatica__stimulation__onion
  :parameters ()
  :precondition (and (fears_sciatica_stimulation) (craves_stimulation_onion) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_sciatica_stimulation)) (craves_sciatica_onion)))
 (:action overcome__angina__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_angina_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_angina_entertainment) (not (craves_angina_beef))))
 (:action overcome__angina__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_angina_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_angina_entertainment) (not (craves_angina_onion))))
 (:action overcome__angina__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_muffin) (craves_angina_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_angina_entertainment) (not (craves_angina_muffin))))
 (:action overcome__laceration__curiosity__tuna
  :parameters ()
  :precondition (and (craves_laceration_tuna) (craves_curiosity_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_laceration_tuna)) (fears_laceration_curiosity)))
 (:action succumb__depression1__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_depression1_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_depression1_ham) (not (fears_depression1_satiety))))
 (:action succumb__depression1__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_depression1_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_depression1_muffin) (not (fears_depression1_satiety))))
 (:action succumb__depression1__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_depression1_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_depression1_beef) (not (fears_depression1_satiety))))
 (:action succumb__depression1__satiety__tuna
  :parameters ()
  :precondition (and (craves_satiety_tuna) (fears_depression1_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_depression1_tuna) (not (fears_depression1_satiety))))
 (:action succumb__depression1__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_depression1_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_depression1_onion) (not (fears_depression1_satiety))))
 (:action succumb__depression1__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_depression1_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_depression1_satiety)) (craves_depression1_cherry)))
 (:action succumb__depression1__satiety__flounder
  :parameters ()
  :precondition (and (craves_satiety_flounder) (fears_depression1_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_depression1_flounder) (not (fears_depression1_satiety))))
 (:action overcome__angina__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v3_4)) (craves_angina_muffin) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_angina_satiety) (not (craves_angina_muffin))))
 (:action succumb__anger__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (fears_anger_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_anger_onion) (not (fears_anger_curiosity))))
 (:action succumb__anger__curiosity__beef
  :parameters ()
  :precondition (and (craves_curiosity_beef) (fears_anger_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_anger_curiosity)) (craves_anger_beef)))
 (:action succumb__anger__curiosity__flounder
  :parameters ()
  :precondition (and (fears_anger_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_anger_curiosity)) (craves_anger_flounder)))
 (:action succumb__anger__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_anger_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_anger_tuna) (not (fears_anger_curiosity))))
 (:action succumb__anger__curiosity__muffin
  :parameters ()
  :precondition (and (craves_curiosity_muffin) (fears_anger_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_anger_muffin) (not (fears_anger_curiosity))))
 (:action succumb__anger__curiosity__cherry
  :parameters ()
  :precondition (and (fears_anger_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_anger_cherry) (not (fears_anger_curiosity))))
 (:action overcome__angina__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (craves_angina_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_angina_beef)) (fears_angina_satiety)))
 (:action overcome__angina__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v3_4)) (craves_angina_cherry) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_angina_satiety) (not (craves_angina_cherry))))
 (:action succumb__dread8__entertainment__onion
  :parameters ()
  :precondition (and (fears_dread8_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_dread8_onion) (not (fears_dread8_entertainment))))
 (:action succumb__dread8__entertainment__beef
  :parameters ()
  :precondition (and (fears_dread8_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread8_entertainment)) (craves_dread8_beef)))
 (:action succumb__dread8__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_dread8_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread8_entertainment)) (craves_dread8_flounder)))
 (:action succumb__jealousy__curiosity__beef
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_jealousy_beef) (not (fears_jealousy_curiosity))))
 (:action overcome__hangover__curiosity__tuna
  :parameters ()
  :precondition (and (craves_hangover_tuna) (craves_curiosity_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_hangover_curiosity) (not (craves_hangover_tuna))))
 (:action succumb__jealousy__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_jealousy_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_jealousy_onion) (not (fears_jealousy_stimulation))))
 (:action succumb__jealousy__stimulation__beef
  :parameters ()
  :precondition (and (craves_stimulation_beef) (fears_jealousy_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_jealousy_stimulation)) (craves_jealousy_beef)))
 (:action succumb__jealousy__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_jealousy_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_jealousy_stimulation)) (craves_jealousy_muffin)))
 (:action succumb__jealousy__stimulation__ham
  :parameters ()
  :precondition (and (craves_stimulation_ham) (fears_jealousy_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_jealousy_stimulation)) (craves_jealousy_ham)))
 (:action overcome__laceration__satiety__tuna
  :parameters ()
  :precondition (and (craves_laceration_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_laceration_satiety) (not (craves_laceration_tuna))))
 (:action succumb__depression1__entertainment__onion
  :parameters ()
  :precondition (and (craves_entertainment_onion) (fears_depression1_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression1_onion) (not (fears_depression1_entertainment))))
 (:action succumb__depression1__entertainment__beef
  :parameters ()
  :precondition (and (craves_entertainment_beef) (fears_depression1_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression1_beef) (not (fears_depression1_entertainment))))
 (:action succumb__depression1__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_depression1_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression1_flounder) (not (fears_depression1_entertainment))))
 (:action succumb__depression1__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_depression1_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression1_tuna) (not (fears_depression1_entertainment))))
 (:action succumb__depression1__entertainment__muffin
  :parameters ()
  :precondition (and (craves_entertainment_muffin) (fears_depression1_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression1_muffin) (not (fears_depression1_entertainment))))
 (:action succumb__depression1__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_depression1_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression1_ham) (not (fears_depression1_entertainment))))
 (:action succumb__prostatitis__satiety__flounder
  :parameters ()
  :precondition (and (fears_prostatitis_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_prostatitis_flounder) (not (fears_prostatitis_satiety))))
 (:action succumb__prostatitis__satiety__tuna
  :parameters ()
  :precondition (and (fears_prostatitis_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis_satiety)) (craves_prostatitis_tuna)))
 (:action succumb__prostatitis__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_prostatitis_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis_satiety)) (craves_prostatitis_muffin)))
 (:action succumb__prostatitis__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_prostatitis_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis_satiety)) (craves_prostatitis_cherry)))
 (:action succumb__prostatitis__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_prostatitis_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_prostatitis_ham) (not (fears_prostatitis_satiety))))
 (:action overcome__jealousy__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (craves_jealousy_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_jealousy_curiosity) (not (craves_jealousy_tuna))))
 (:action overcome__jealousy__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_jealousy_cherry) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy_cherry)) (fears_jealousy_curiosity)))
 (:action succumb__prostatitis__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_prostatitis_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis_satiety)) (craves_prostatitis_onion)))
 (:action succumb__prostatitis__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_prostatitis_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis_satiety)) (craves_prostatitis_beef)))
 (:action overcome__jealousy__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_jealousy_tuna) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_jealousy_stimulation) (not (craves_jealousy_tuna))))
 (:action overcome__jealousy__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_jealousy_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_jealousy_cherry)) (fears_jealousy_stimulation)))
 (:action succumb__abrasion__curiosity__flounder
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_abrasion_flounder) (not (fears_abrasion_curiosity))))
 (:action succumb__abrasion__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_abrasion_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_tuna)))
 (:action succumb__abrasion__curiosity__muffin
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_abrasion_muffin) (not (fears_abrasion_curiosity))))
 (:action succumb__abrasion__curiosity__cherry
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_abrasion_cherry) (not (fears_abrasion_curiosity))))
 (:action overcome__hangover__satiety__tuna
  :parameters ()
  :precondition (and (craves_hangover_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_hangover_tuna)) (fears_hangover_satiety)))
 (:action succumb__abrasion__curiosity__onion
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_abrasion_onion) (not (fears_abrasion_curiosity))))
 (:action succumb__abrasion__curiosity__beef
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_beef)))
 (:action overcome__jealousy__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_jealousy_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy_entertainment) (not (craves_jealousy_tuna))))
 (:action overcome__jealousy__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_jealousy_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy_entertainment) (not (craves_jealousy_flounder))))
 (:action succumb__anger__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_anger_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anger_satiety)) (craves_anger_muffin)))
 (:action succumb__anger__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_anger_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anger_satiety)) (craves_anger_cherry)))
 (:action succumb__anger__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_anger_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anger_satiety)) (craves_anger_ham)))
 (:action succumb__anger__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_anger_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anger_satiety)) (craves_anger_onion)))
 (:action succumb__anger__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_anger_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anger_satiety)) (craves_anger_beef)))
 (:action overcome__depression1__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_depression1_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_depression1_curiosity) (not (craves_depression1_muffin))))
 (:action succumb__anger__satiety__flounder
  :parameters ()
  :precondition (and (fears_anger_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anger_satiety)) (craves_anger_flounder)))
 (:action succumb__anger__satiety__tuna
  :parameters ()
  :precondition (and (fears_anger_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anger_satiety)) (craves_anger_tuna)))
 (:action overcome__depression1__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_depression1_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_depression1_muffin)) (fears_depression1_stimulation)))
 (:action overcome__anger__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_anger_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anger_entertainment) (not (craves_anger_beef))))
 (:action succumb__prostatitis__curiosity__muffin
  :parameters ()
  :precondition (and (fears_prostatitis_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_prostatitis_muffin) (not (fears_prostatitis_curiosity))))
 (:action succumb__prostatitis__curiosity__cherry
  :parameters ()
  :precondition (and (fears_prostatitis_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_prostatitis_curiosity)) (craves_prostatitis_cherry)))
 (:action overcome__jealousy__satiety__tuna
  :parameters ()
  :precondition (and (craves_jealousy_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_jealousy_satiety) (not (craves_jealousy_tuna))))
 (:action overcome__jealousy__satiety__flounder
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_flounder) (craves_jealousy_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_jealousy_satiety) (not (craves_jealousy_flounder))))
 (:action succumb__prostatitis__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (fears_prostatitis_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_prostatitis_onion) (not (fears_prostatitis_curiosity))))
 (:action overcome__jealousy__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_jealousy_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_jealousy_cherry)) (fears_jealousy_satiety)))
 (:action succumb__prostatitis__curiosity__beef
  :parameters ()
  :precondition (and (fears_prostatitis_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_prostatitis_curiosity)) (craves_prostatitis_beef)))
 (:action succumb__prostatitis__curiosity__flounder
  :parameters ()
  :precondition (and (fears_prostatitis_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_prostatitis_flounder) (not (fears_prostatitis_curiosity))))
 (:action succumb__prostatitis__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_prostatitis_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_prostatitis_tuna) (not (fears_prostatitis_curiosity))))
 (:action succumb__boils4__entertainment__onion
  :parameters ()
  :precondition (and (fears_boils4_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils4_onion) (not (fears_boils4_entertainment))))
 (:action succumb__boils4__entertainment__beef
  :parameters ()
  :precondition (and (fears_boils4_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_boils4_entertainment)) (craves_boils4_beef)))
 (:action succumb__boils4__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_boils4_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils4_flounder) (not (fears_boils4_entertainment))))
 (:action overcome__dread8__curiosity__ham
  :parameters ()
  :precondition (and (craves_dread8_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread8_ham)) (fears_dread8_curiosity)))
 (:action succumb__angina__curiosity__ham
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_ham)))
 (:action overcome__dread8__curiosity__beef
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_beef) (craves_dread8_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread8_beef)) (fears_dread8_curiosity)))
 (:action overcome__dread8__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_dread8_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread8_onion)) (fears_dread8_curiosity)))
 (:action overcome__dread8__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_dread8_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread8_flounder)) (fears_dread8_curiosity)))
 (:action overcome__anger__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_anger_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_anger_satiety) (not (craves_anger_cherry))))
 (:action overcome__anger__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (craves_anger_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_anger_satiety) (not (craves_anger_muffin))))
 (:action overcome__anger__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (craves_anger_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_anger_satiety) (not (craves_anger_ham))))
 (:action succumb__angina__stimulation__ham
  :parameters ()
  :precondition (and (fears_angina_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_angina_stimulation)) (craves_angina_ham)))
 (:action overcome__anger__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v3_4)) (craves_anger_onion) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_anger_satiety) (not (craves_anger_onion))))
 (:action overcome__anger__satiety__tuna
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_tuna) (craves_anger_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_anger_satiety) (not (craves_anger_tuna))))
 (:action overcome__anger__satiety__flounder
  :parameters ()
  :precondition (and (craves_anger_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_anger_flounder)) (fears_anger_satiety)))
 (:action succumb__angina__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_angina_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_angina_onion) (not (fears_angina_stimulation))))
 (:action succumb__angina__stimulation__beef
  :parameters ()
  :precondition (and (fears_angina_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_angina_stimulation)) (craves_angina_beef)))
 (:action succumb__angina__stimulation__flounder
  :parameters ()
  :precondition (and (fears_angina_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_angina_stimulation)) (craves_angina_flounder)))
 (:action succumb__angina__stimulation__tuna
  :parameters ()
  :precondition (and (fears_angina_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_angina_stimulation)) (craves_angina_tuna)))
 (:action succumb__angina__stimulation__muffin
  :parameters ()
  :precondition (and (fears_angina_stimulation) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_angina_stimulation)) (craves_angina_muffin)))
 (:action succumb__angina__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_angina_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_angina_stimulation)) (craves_angina_cherry)))
 (:action overcome__dread8__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_beef) (craves_dread8_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread8_stimulation) (not (craves_dread8_beef))))
 (:action overcome__dread8__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_dread8_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread8_stimulation) (not (craves_dread8_onion))))
 (:action overcome__dread8__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_dread8_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread8_stimulation) (not (craves_dread8_flounder))))
 (:action overcome__prostatitis__entertainment__muffin
  :parameters ()
  :precondition (and (craves_prostatitis_muffin) (not (v0_4)) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis_entertainment) (not (craves_prostatitis_muffin))))
 (:action overcome__prostatitis__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_prostatitis_ham) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis_entertainment) (not (craves_prostatitis_ham))))
 (:action overcome__prostatitis__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_prostatitis_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis_entertainment) (not (craves_prostatitis_onion))))
 (:action overcome__prostatitis__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_prostatitis_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis_entertainment) (not (craves_prostatitis_tuna))))
 (:action overcome__prostatitis__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_prostatitis_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis_entertainment) (not (craves_prostatitis_flounder))))
 (:action overcome__prostatitis__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_prostatitis_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis_entertainment) (not (craves_prostatitis_cherry))))
 (:action overcome__depression1__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (craves_depression1_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_depression1_satiety) (not (craves_depression1_ham))))
 (:action overcome__depression1__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (craves_depression1_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression1_beef)) (fears_depression1_satiety)))
 (:action overcome__depression1__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v3_4)) (craves_depression1_onion) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_depression1_satiety) (not (craves_depression1_onion))))
 (:action overcome__depression1__satiety__tuna
  :parameters ()
  :precondition (and (craves_depression1_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression1_tuna)) (fears_depression1_satiety)))
 (:action overcome__depression1__satiety__flounder
  :parameters ()
  :precondition (and (craves_depression1_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression1_flounder)) (fears_depression1_satiety)))
 (:action overcome__depression1__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_depression1_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression1_cherry)) (fears_depression1_satiety)))
 (:action overcome__anger__curiosity__ham
  :parameters ()
  :precondition (and (craves_anger_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_anger_curiosity) (not (craves_anger_ham))))
 (:action overcome__anger__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_anger_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_anger_curiosity) (not (craves_anger_onion))))
 (:action overcome__anger__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_anger_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_anger_tuna)) (fears_anger_curiosity)))
 (:action overcome__anger__curiosity__flounder
  :parameters ()
  :precondition (and (craves_anger_flounder) (not (v5_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_anger_flounder)) (fears_anger_curiosity)))
 (:action overcome__anger__curiosity__cherry
  :parameters ()
  :precondition (and (craves_anger_cherry) (not (v5_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_anger_curiosity) (not (craves_anger_cherry))))
 (:action overcome__anger__curiosity__muffin
  :parameters ()
  :precondition (and (craves_anger_muffin) (not (v5_4)) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_anger_muffin)) (fears_anger_curiosity)))
 (:action succumb__laceration__entertainment__cherry
  :parameters ()
  :precondition (and (fears_laceration_entertainment) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_laceration_entertainment)) (craves_laceration_cherry)))
 (:action overcome__dread8__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_dread8_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread8_entertainment) (not (craves_dread8_beef))))
 (:action overcome__dread8__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_dread8_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread8_entertainment) (not (craves_dread8_onion))))
 (:action overcome__dread8__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_dread8_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread8_entertainment) (not (craves_dread8_flounder))))
 (:action overcome__anger__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_anger_muffin) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_anger_stimulation) (not (craves_anger_muffin))))
 (:action overcome__anger__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_anger_ham) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_anger_stimulation) (not (craves_anger_ham))))
 (:action succumb__laceration__curiosity__ham
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_laceration_ham) (not (fears_laceration_curiosity))))
 (:action succumb__laceration__curiosity__muffin
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_laceration_curiosity)) (craves_laceration_muffin)))
 (:action overcome__anger__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_anger_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_anger_stimulation) (not (craves_anger_onion))))
 (:action overcome__anger__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_tuna) (craves_anger_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_anger_stimulation) (not (craves_anger_tuna))))
 (:action overcome__anger__stimulation__flounder
  :parameters ()
  :precondition (and (craves_anger_flounder) (not (v4_4)) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_anger_flounder)) (fears_anger_stimulation)))
 (:action overcome__anger__stimulation__cherry
  :parameters ()
  :precondition (and (craves_anger_cherry) (craves_stimulation_cherry) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_anger_stimulation) (not (craves_anger_cherry))))
 (:action overcome__boils4__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_boils4_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils4_flounder)) (fears_boils4_curiosity)))
 (:action overcome__boils4__curiosity__ham
  :parameters ()
  :precondition (and (craves_boils4_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils4_ham)) (fears_boils4_curiosity)))
 (:action succumb__loneliness__curiosity__ham
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_loneliness_ham) (not (fears_loneliness_curiosity))))
 (:action overcome__boils4__curiosity__beef
  :parameters ()
  :precondition (and (craves_boils4_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_boils4_curiosity) (not (craves_boils4_beef))))
 (:action overcome__boils4__curiosity__onion
  :parameters ()
  :precondition (and (craves_boils4_onion) (craves_curiosity_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils4_onion)) (fears_boils4_curiosity)))
 (:action overcome__prostatitis__satiety__tuna
  :parameters ()
  :precondition (and (not (v3_4)) (craves_prostatitis_tuna) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_prostatitis_satiety) (not (craves_prostatitis_tuna))))
 (:action overcome__prostatitis__satiety__flounder
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_flounder) (craves_prostatitis_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_prostatitis_satiety) (not (craves_prostatitis_flounder))))
 (:action overcome__prostatitis__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v3_4)) (craves_prostatitis_cherry) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_prostatitis_satiety) (not (craves_prostatitis_cherry))))
 (:action overcome__prostatitis__satiety__muffin
  :parameters ()
  :precondition (and (craves_prostatitis_muffin) (craves_satiety_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_prostatitis_satiety) (not (craves_prostatitis_muffin))))
 (:action overcome__prostatitis__satiety__ham
  :parameters ()
  :precondition (and (craves_prostatitis_ham) (craves_satiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_prostatitis_ham)) (fears_prostatitis_satiety)))
 (:action overcome__prostatitis__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (craves_prostatitis_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_prostatitis_satiety) (not (craves_prostatitis_onion))))
 (:action succumb__loneliness__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_loneliness_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_loneliness_stimulation)) (craves_loneliness_onion)))
 (:action overcome__boils4__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_boils4_flounder) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils4_flounder)) (fears_boils4_stimulation)))
 (:action succumb__loneliness__stimulation__beef
  :parameters ()
  :precondition (and (fears_loneliness_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_loneliness_stimulation)) (craves_loneliness_beef)))
 (:action succumb__loneliness__stimulation__flounder
  :parameters ()
  :precondition (and (fears_loneliness_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_loneliness_flounder) (not (fears_loneliness_stimulation))))
 (:action succumb__loneliness__stimulation__tuna
  :parameters ()
  :precondition (and (fears_loneliness_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_loneliness_stimulation)) (craves_loneliness_tuna)))
 (:action succumb__loneliness__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_loneliness_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_loneliness_muffin) (not (fears_loneliness_stimulation))))
 (:action succumb__loneliness__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_loneliness_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_loneliness_cherry) (not (fears_loneliness_stimulation))))
 (:action succumb__loneliness__stimulation__ham
  :parameters ()
  :precondition (and (fears_loneliness_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_loneliness_ham) (not (fears_loneliness_stimulation))))
 (:action overcome__boils4__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_boils4_beef) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_boils4_stimulation) (not (craves_boils4_beef))))
 (:action overcome__boils4__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (craves_boils4_onion) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils4_onion)) (fears_boils4_stimulation)))
 (:action overcome__abrasion__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_abrasion_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_abrasion_entertainment) (not (craves_abrasion_flounder))))
 (:action overcome__abrasion__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_abrasion_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_abrasion_entertainment) (not (craves_abrasion_cherry))))
 (:action overcome__abrasion__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_abrasion_muffin) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_abrasion_entertainment) (not (craves_abrasion_muffin))))
 (:action overcome__abrasion__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_abrasion_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_abrasion_entertainment) (not (craves_abrasion_ham))))
 (:action overcome__abrasion__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_onion) (craves_abrasion_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_abrasion_entertainment) (not (craves_abrasion_onion))))
 (:action overcome__abrasion__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_abrasion_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_abrasion_tuna)) (fears_abrasion_entertainment)))
 (:action overcome__dread8__satiety__flounder
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_flounder) (craves_dread8_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_dread8_satiety) (not (craves_dread8_flounder))))
 (:action overcome__dread8__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (not (v3_4)) (craves_dread8_beef) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread8_beef)) (fears_dread8_satiety)))
 (:action overcome__dread8__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (craves_dread8_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread8_onion)) (fears_dread8_satiety)))
 (:action overcome__prostatitis__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_cherry) (craves_prostatitis_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_prostatitis_curiosity) (not (craves_prostatitis_cherry))))
 (:action overcome__prostatitis__curiosity__muffin
  :parameters ()
  :precondition (and (craves_prostatitis_muffin) (not (v5_4)) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_prostatitis_curiosity) (not (craves_prostatitis_muffin))))
 (:action overcome__prostatitis__curiosity__ham
  :parameters ()
  :precondition (and (craves_prostatitis_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_prostatitis_ham)) (fears_prostatitis_curiosity)))
 (:action overcome__prostatitis__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_prostatitis_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_prostatitis_curiosity) (not (craves_prostatitis_onion))))
 (:action overcome__prostatitis__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_prostatitis_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_prostatitis_tuna)) (fears_prostatitis_curiosity)))
 (:action overcome__prostatitis__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_prostatitis_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_prostatitis_flounder)) (fears_prostatitis_curiosity)))
 (:action succumb__angina__entertainment__onion
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_angina_onion) (not (fears_angina_entertainment))))
 (:action succumb__angina__entertainment__beef
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_beef)))
 (:action succumb__angina__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_angina_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_flounder)))
 (:action succumb__angina__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_angina_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_tuna)))
 (:action succumb__angina__entertainment__muffin
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_muffin)))
 (:action succumb__angina__entertainment__cherry
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_cherry)))
 (:action succumb__angina__entertainment__ham
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_ham)))
 (:action overcome__boils4__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_boils4_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_boils4_entertainment) (not (craves_boils4_beef))))
 (:action overcome__boils4__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_boils4_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_boils4_entertainment) (not (craves_boils4_onion))))
 (:action overcome__boils4__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_boils4_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_boils4_flounder)) (fears_boils4_entertainment)))
 (:action overcome__prostatitis__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_prostatitis_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_prostatitis_stimulation) (not (craves_prostatitis_flounder))))
 (:action overcome__prostatitis__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_prostatitis_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_prostatitis_stimulation) (not (craves_prostatitis_cherry))))
 (:action overcome__prostatitis__stimulation__muffin
  :parameters ()
  :precondition (and (craves_prostatitis_muffin) (not (v4_4)) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_prostatitis_stimulation) (not (craves_prostatitis_muffin))))
 (:action overcome__prostatitis__stimulation__ham
  :parameters ()
  :precondition (and (craves_prostatitis_ham) (not (v4_4)) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_prostatitis_stimulation) (not (craves_prostatitis_ham))))
 (:action overcome__prostatitis__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_prostatitis_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_prostatitis_stimulation) (not (craves_prostatitis_onion))))
 (:action overcome__prostatitis__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_prostatitis_tuna) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_prostatitis_stimulation) (not (craves_prostatitis_tuna))))
 (:action succumb__hangover__curiosity__beef
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_hangover_curiosity)) (craves_hangover_beef)))
 (:action succumb__hangover__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_hangover_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_hangover_tuna) (not (fears_hangover_curiosity))))
 (:action succumb__hangover__curiosity__onion
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_hangover_curiosity)) (craves_hangover_onion)))
 (:action overcome__abrasion__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v3_4)) (craves_abrasion_onion) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_abrasion_satiety) (not (craves_abrasion_onion))))
 (:action overcome__abrasion__satiety__tuna
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_tuna) (craves_abrasion_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_abrasion_tuna)) (fears_abrasion_satiety)))
 (:action overcome__abrasion__satiety__flounder
  :parameters ()
  :precondition (and (craves_abrasion_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_abrasion_satiety) (not (craves_abrasion_flounder))))
 (:action overcome__abrasion__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_abrasion_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_abrasion_satiety) (not (craves_abrasion_cherry))))
 (:action overcome__abrasion__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v3_4)) (craves_abrasion_muffin) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_abrasion_satiety) (not (craves_abrasion_muffin))))
 (:action overcome__abrasion__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (not (v3_4)) (craves_abrasion_ham) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_abrasion_satiety) (not (craves_abrasion_ham))))
 (:action overcome__dread__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_dread_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread_entertainment) (not (craves_dread_tuna))))
 (:action overcome__dread__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_dread_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread_entertainment) (not (craves_dread_flounder))))
 (:action overcome__dread__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_dread_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread_entertainment) (not (craves_dread_cherry))))
 (:action overcome__dread__entertainment__ham
  :parameters ()
  :precondition (and (craves_dread_ham) (not (v0_4)) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread_entertainment) (not (craves_dread_ham))))
 (:action succumb__hangover__satiety__tuna
  :parameters ()
  :precondition (and (fears_hangover_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_hangover_tuna) (not (fears_hangover_satiety))))
 (:action succumb__hangover__satiety__onion
  :parameters ()
  :precondition (and (fears_hangover_satiety) (craves_satiety_onion) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_hangover_onion) (not (fears_hangover_satiety))))
 (:action succumb__hangover__satiety__cherry
  :parameters ()
  :precondition (and (fears_hangover_satiety) (craves_satiety_cherry) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_hangover_cherry) (not (fears_hangover_satiety))))
 (:action succumb__hangover__satiety__flounder
  :parameters ()
  :precondition (and (fears_hangover_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_hangover_flounder) (not (fears_hangover_satiety))))
 (:action succumb__hangover__satiety__ham
  :parameters ()
  :precondition (and (fears_hangover_satiety) (craves_satiety_ham) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_hangover_ham) (not (fears_hangover_satiety))))
 (:action succumb__hangover__satiety__muffin
  :parameters ()
  :precondition (and (fears_hangover_satiety) (craves_satiety_muffin) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_hangover_muffin) (not (fears_hangover_satiety))))
 (:action overcome__boils4__satiety__onion
  :parameters ()
  :precondition (and (craves_boils4_onion) (craves_satiety_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_boils4_satiety) (not (craves_boils4_onion))))
 (:action overcome__boils4__satiety__flounder
  :parameters ()
  :precondition (and (craves_boils4_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_boils4_flounder)) (fears_boils4_satiety)))
 (:action succumb__hangover__satiety__beef
  :parameters ()
  :precondition (and (fears_hangover_satiety) (craves_satiety_beef) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_hangover_beef) (not (fears_hangover_satiety))))
 (:action overcome__boils4__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (craves_boils4_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_boils4_satiety) (not (craves_boils4_beef))))
 (:action overcome__abrasion__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_abrasion_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_abrasion_tuna)) (fears_abrasion_curiosity)))
 (:action overcome__abrasion__curiosity__flounder
  :parameters ()
  :precondition (and (craves_abrasion_flounder) (not (v5_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_flounder))))
 (:action overcome__abrasion__curiosity__cherry
  :parameters ()
  :precondition (and (craves_abrasion_cherry) (not (v5_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_cherry))))
 (:action overcome__abrasion__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_abrasion_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_abrasion_muffin)) (fears_abrasion_curiosity)))
 (:action overcome__abrasion__curiosity__ham
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_ham) (craves_abrasion_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_ham))))
 (:action succumb__jealousy__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_flounder)))
 (:action succumb__jealousy__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_tuna)))
 (:action succumb__jealousy__entertainment__muffin
  :parameters ()
  :precondition (and (craves_entertainment_muffin) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_muffin)))
 (:action succumb__jealousy__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_cherry)))
 (:action succumb__jealousy__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_ham)))
 (:action overcome__abrasion__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_abrasion_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_abrasion_onion)) (fears_abrasion_curiosity)))
 (:action feast__curiosity__ham__tuna
  :parameters ()
  :precondition (and (craves_curiosity_ham) (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (craves_curiosity_tuna) (not (craves_curiosity_ham))))
 (:action succumb__jealousy__entertainment__onion
  :parameters ()
  :precondition (and (craves_entertainment_onion) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_onion)))
 (:action succumb__jealousy__entertainment__beef
  :parameters ()
  :precondition (and (craves_entertainment_beef) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_beef)))
 (:action feast__curiosity__ham__muffin
  :parameters ()
  :precondition (and (craves_curiosity_ham) (not (v9_4)) (not (of)))
  :effect (and (when (z_9_0) (v9_0)) (when (not (z_9_0)) (not (v9_0))) (when (z_9_1) (v9_1)) (when (not (z_9_1)) (not (v9_1))) (when (z_9_2) (v9_2)) (when (not (z_9_2)) (not (v9_2))) (when (z_9_3) (v9_3)) (when (not (z_9_3)) (not (v9_3))) (when (z_9_4) (v9_4)) (when (not (z_9_4)) (not (v9_4))) (when (and (v9_4) (not (z_9_4))) (of)) (not (craves_curiosity_ham)) (craves_curiosity_muffin)))
 (:action overcome__abrasion__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_abrasion_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_abrasion_stimulation) (not (craves_abrasion_onion))))
 (:action overcome__abrasion__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_tuna) (craves_abrasion_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_abrasion_tuna)) (fears_abrasion_stimulation)))
 (:action overcome__abrasion__stimulation__flounder
  :parameters ()
  :precondition (and (craves_abrasion_flounder) (not (v4_4)) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_abrasion_stimulation) (not (craves_abrasion_flounder))))
 (:action overcome__abrasion__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_abrasion_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_abrasion_stimulation) (not (craves_abrasion_cherry))))
 (:action overcome__abrasion__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_abrasion_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_abrasion_stimulation) (not (craves_abrasion_muffin))))
 (:action overcome__abrasion__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_ham) (craves_abrasion_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_abrasion_stimulation) (not (craves_abrasion_ham))))
 (:action succumb__jealousy__curiosity__ham
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_jealousy_ham) (not (fears_jealousy_curiosity))))
 (:action succumb__laceration__curiosity__beef
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_laceration_beef) (not (fears_laceration_curiosity))))
 (:action succumb__laceration__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_laceration_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_laceration_tuna) (not (fears_laceration_curiosity))))
 (:action succumb__laceration__curiosity__onion
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_laceration_curiosity)) (craves_laceration_onion)))
 (:action succumb__laceration__curiosity__cherry
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_laceration_curiosity)) (craves_laceration_cherry)))
 (:action succumb__laceration__curiosity__flounder
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_laceration_flounder) (not (fears_laceration_curiosity))))
 (:action overcome__dread__satiety__ham
  :parameters ()
  :precondition (and (craves_dread_ham) (craves_satiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread_ham)) (fears_dread_satiety)))
 (:action overcome__dread__satiety__tuna
  :parameters ()
  :precondition (and (craves_dread_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread_tuna)) (fears_dread_satiety)))
 (:action overcome__dread__satiety__flounder
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_flounder) (craves_dread_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_dread_flounder)) (fears_dread_satiety)))
 (:action overcome__boils__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_boils_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_boils_entertainment) (not (craves_boils_beef))))
 (:action overcome__boils__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_boils_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_boils_onion)) (fears_boils_entertainment)))
 (:action overcome__boils__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_boils_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_boils_entertainment) (not (craves_boils_flounder))))
 (:action overcome__boils__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_boils_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_boils_cherry)) (fears_boils_entertainment)))
 (:action overcome__boils__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_muffin) (craves_boils_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_boils_entertainment) (not (craves_boils_muffin))))
 (:action overcome__boils__entertainment__ham
  :parameters ()
  :precondition (and (craves_boils_ham) (not (v0_4)) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_boils_entertainment) (not (craves_boils_ham))))
 (:action succumb__laceration__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_laceration_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_laceration_satiety)) (craves_laceration_cherry)))
 (:action succumb__laceration__satiety__flounder
  :parameters ()
  :precondition (and (fears_laceration_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_laceration_satiety)) (craves_laceration_flounder)))
 (:action succumb__laceration__satiety__ham
  :parameters ()
  :precondition (and (fears_laceration_satiety) (craves_satiety_ham) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_laceration_satiety)) (craves_laceration_ham)))
 (:action succumb__laceration__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_laceration_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_laceration_satiety)) (craves_laceration_muffin)))
 (:action succumb__laceration__satiety__beef
  :parameters ()
  :precondition (and (fears_laceration_satiety) (craves_satiety_beef) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_laceration_beef) (not (fears_laceration_satiety))))
 (:action succumb__laceration__satiety__tuna
  :parameters ()
  :precondition (and (fears_laceration_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_laceration_tuna) (not (fears_laceration_satiety))))
 (:action succumb__laceration__satiety__onion
  :parameters ()
  :precondition (and (fears_laceration_satiety) (craves_satiety_onion) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_laceration_satiety)) (craves_laceration_onion)))
 (:action overcome__dread__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_dread_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread_tuna)) (fears_dread_curiosity)))
 (:action overcome__dread__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_dread_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread_flounder)) (fears_dread_curiosity)))
 (:action overcome__dread__curiosity__ham
  :parameters ()
  :precondition (and (craves_dread_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread_ham)) (fears_dread_curiosity)))
 (:action succumb__hangover__entertainment__cherry
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_hangover_cherry) (not (fears_hangover_entertainment))))
 (:action overcome__dread__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_dread_tuna) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread_stimulation) (not (craves_dread_tuna))))
 (:action overcome__dread__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_dread_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread_stimulation) (not (craves_dread_flounder))))
 (:action overcome__dread__stimulation__ham
  :parameters ()
  :precondition (and (craves_dread_ham) (not (v4_4)) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread_stimulation) (not (craves_dread_ham))))
 (:action succumb__hangover__curiosity__cherry
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_hangover_cherry) (not (fears_hangover_curiosity))))
 (:action succumb__hangover__curiosity__flounder
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_hangover_curiosity)) (craves_hangover_flounder)))
 (:action succumb__hangover__curiosity__ham
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_hangover_ham) (not (fears_hangover_curiosity))))
 (:action succumb__hangover__curiosity__muffin
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_hangover_curiosity)) (craves_hangover_muffin)))
 (:action overcome__boils__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v3_4)) (craves_boils_cherry) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_boils_cherry)) (fears_boils_satiety)))
 (:action overcome__boils__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v3_4)) (craves_boils_muffin) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_boils_satiety) (not (craves_boils_muffin))))
 (:action overcome__boils__satiety__ham
  :parameters ()
  :precondition (and (craves_boils_ham) (craves_satiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_boils_ham)) (fears_boils_satiety)))
 (:action overcome__boils__satiety__beef
  :parameters ()
  :precondition (and (craves_boils_beef) (craves_satiety_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_boils_satiety) (not (craves_boils_beef))))
 (:action overcome__boils__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (craves_boils_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_boils_onion)) (fears_boils_satiety)))
 (:action overcome__boils__satiety__flounder
  :parameters ()
  :precondition (and (craves_boils_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_boils_flounder)) (fears_boils_satiety)))
 (:action succumb__grief7__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_grief7_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_grief7_entertainment)) (craves_grief7_cherry)))
 (:action overcome__sciatica__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_muffin) (craves_sciatica_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_sciatica_entertainment) (not (craves_sciatica_muffin))))
 (:action overcome__sciatica__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_sciatica_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_sciatica_ham)) (fears_sciatica_entertainment)))
 (:action succumb__grief7__curiosity__ham
  :parameters ()
  :precondition (and (craves_curiosity_ham) (fears_grief7_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_grief7_curiosity)) (craves_grief7_ham)))
 (:action overcome__sciatica__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_sciatica_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_sciatica_beef)) (fears_sciatica_entertainment)))
 (:action overcome__sciatica__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_sciatica_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_sciatica_onion)) (fears_sciatica_entertainment)))
 (:action overcome__sciatica__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_sciatica_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_sciatica_entertainment) (not (craves_sciatica_flounder))))
 (:action overcome__sciatica__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_sciatica_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_sciatica_cherry)) (fears_sciatica_entertainment)))
 (:action succumb__grief7__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (fears_grief7_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_grief7_onion) (not (fears_grief7_curiosity))))
 (:action succumb__grief7__curiosity__beef
  :parameters ()
  :precondition (and (craves_curiosity_beef) (fears_grief7_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_grief7_curiosity)) (craves_grief7_beef)))
 (:action succumb__grief7__curiosity__flounder
  :parameters ()
  :precondition (and (fears_grief7_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_grief7_flounder) (not (fears_grief7_curiosity))))
 (:action succumb__grief7__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_grief7_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_grief7_curiosity)) (craves_grief7_tuna)))
 (:action succumb__grief7__curiosity__muffin
  :parameters ()
  :precondition (and (craves_curiosity_muffin) (fears_grief7_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_grief7_muffin) (not (fears_grief7_curiosity))))
 (:action succumb__grief7__curiosity__cherry
  :parameters ()
  :precondition (and (fears_grief7_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_grief7_cherry) (not (fears_grief7_curiosity))))
 (:action overcome__boils__curiosity__ham
  :parameters ()
  :precondition (and (craves_boils_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils_ham)) (fears_boils_curiosity)))
 (:action succumb__grief7__stimulation__ham
  :parameters ()
  :precondition (and (craves_stimulation_ham) (fears_grief7_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_grief7_stimulation)) (craves_grief7_ham)))
 (:action succumb__grief7__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_grief7_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_grief7_muffin) (not (fears_grief7_stimulation))))
 (:action overcome__boils__curiosity__beef
  :parameters ()
  :precondition (and (craves_boils_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_boils_curiosity) (not (craves_boils_beef))))
 (:action overcome__boils__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (craves_boils_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils_onion)) (fears_boils_curiosity)))
 (:action overcome__boils__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_boils_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils_flounder)) (fears_boils_curiosity)))
 (:action overcome__boils__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_boils_cherry) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils_cherry)) (fears_boils_curiosity)))
 (:action overcome__boils__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_boils_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_boils_curiosity) (not (craves_boils_muffin))))
 (:action succumb__grief7__stimulation__beef
  :parameters ()
  :precondition (and (craves_stimulation_beef) (fears_grief7_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_grief7_stimulation)) (craves_grief7_beef)))
 (:action succumb__grief7__stimulation__tuna
  :parameters ()
  :precondition (and (fears_grief7_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_grief7_stimulation)) (craves_grief7_tuna)))
 (:action succumb__grief7__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_grief7_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_grief7_onion) (not (fears_grief7_stimulation))))
 (:action succumb__grief7__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_grief7_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_grief7_cherry) (not (fears_grief7_stimulation))))
 (:action succumb__grief7__stimulation__flounder
  :parameters ()
  :precondition (and (craves_stimulation_flounder) (fears_grief7_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_grief7_flounder) (not (fears_grief7_stimulation))))
 (:action succumb__depression__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_depression_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression_ham) (not (fears_depression_entertainment))))
 (:action overcome__boils__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_boils_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_boils_stimulation) (not (craves_boils_muffin))))
 (:action overcome__boils__stimulation__ham
  :parameters ()
  :precondition (and (craves_boils_ham) (not (v4_4)) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils_ham)) (fears_boils_stimulation)))
 (:action succumb__depression__entertainment__onion
  :parameters ()
  :precondition (and (fears_depression_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression_onion) (not (fears_depression_entertainment))))
 (:action succumb__depression__entertainment__beef
  :parameters ()
  :precondition (and (fears_depression_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression_beef) (not (fears_depression_entertainment))))
 (:action succumb__depression__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_depression_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression_flounder) (not (fears_depression_entertainment))))
 (:action succumb__depression__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_depression_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression_tuna) (not (fears_depression_entertainment))))
 (:action succumb__depression__entertainment__muffin
  :parameters ()
  :precondition (and (fears_depression_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression_muffin) (not (fears_depression_entertainment))))
 (:action succumb__depression__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_depression_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_depression_cherry) (not (fears_depression_entertainment))))
 (:action succumb__grief7__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_grief7_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_grief7_muffin) (not (fears_grief7_satiety))))
 (:action succumb__grief7__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_grief7_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_grief7_cherry) (not (fears_grief7_satiety))))
 (:action overcome__boils__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_boils_beef) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils_beef)) (fears_boils_stimulation)))
 (:action succumb__grief7__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_grief7_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_grief7_satiety)) (craves_grief7_ham)))
 (:action overcome__boils__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_boils_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils_onion)) (fears_boils_stimulation)))
 (:action overcome__boils__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_boils_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils_flounder)) (fears_boils_stimulation)))
 (:action overcome__boils__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_boils_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils_cherry)) (fears_boils_stimulation)))
 (:action succumb__grief7__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_grief7_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_grief7_onion) (not (fears_grief7_satiety))))
 (:action succumb__grief7__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_grief7_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_grief7_satiety)) (craves_grief7_beef)))
 (:action succumb__grief7__satiety__flounder
  :parameters ()
  :precondition (and (fears_grief7_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_grief7_flounder) (not (fears_grief7_satiety))))
 (:action succumb__grief7__satiety__tuna
  :parameters ()
  :precondition (and (fears_grief7_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_grief7_satiety)) (craves_grief7_tuna)))
 (:action overcome__sciatica__satiety__flounder
  :parameters ()
  :precondition (and (craves_sciatica_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_sciatica_satiety) (not (craves_sciatica_flounder))))
 (:action overcome__sciatica__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_sciatica_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_sciatica_cherry)) (fears_sciatica_satiety)))
 (:action overcome__sciatica__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v3_4)) (craves_sciatica_muffin) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_sciatica_satiety) (not (craves_sciatica_muffin))))
 (:action overcome__sciatica__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (not (v3_4)) (craves_sciatica_ham) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_sciatica_ham)) (fears_sciatica_satiety)))
 (:action overcome__sciatica__satiety__beef
  :parameters ()
  :precondition (and (craves_sciatica_beef) (craves_satiety_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_sciatica_beef)) (fears_sciatica_satiety)))
 (:action overcome__sciatica__satiety__onion
  :parameters ()
  :precondition (and (craves_sciatica_onion) (craves_satiety_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_sciatica_onion)) (fears_sciatica_satiety)))
 (:action succumb__jealousy2__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_jealousy2_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy2_entertainment)) (craves_jealousy2_tuna)))
 (:action succumb__jealousy2__entertainment__onion
  :parameters ()
  :precondition (and (craves_entertainment_onion) (fears_jealousy2_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_jealousy2_onion) (not (fears_jealousy2_entertainment))))
 (:action succumb__jealousy2__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_jealousy2_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy2_entertainment)) (craves_jealousy2_cherry)))
 (:action succumb__jealousy2__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_jealousy2_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy2_entertainment)) (craves_jealousy2_flounder)))
 (:action succumb__jealousy2__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_jealousy2_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy2_entertainment)) (craves_jealousy2_ham)))
 (:action succumb__jealousy2__entertainment__muffin
  :parameters ()
  :precondition (and (craves_entertainment_muffin) (fears_jealousy2_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy2_entertainment)) (craves_jealousy2_muffin)))
 (:action succumb__jealousy2__entertainment__beef
  :parameters ()
  :precondition (and (craves_entertainment_beef) (fears_jealousy2_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_jealousy2_entertainment)) (craves_jealousy2_beef)))
 (:action overcome__anxiety__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_anxiety_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anxiety_entertainment) (not (craves_anxiety_cherry))))
 (:action overcome__anxiety__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_muffin) (craves_anxiety_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_anxiety_muffin)) (fears_anxiety_entertainment)))
 (:action overcome__anxiety__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_anxiety_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anxiety_entertainment) (not (craves_anxiety_ham))))
 (:action overcome__anxiety__entertainment__beef
  :parameters ()
  :precondition (and (craves_anxiety_beef) (not (v0_4)) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anxiety_entertainment) (not (craves_anxiety_beef))))
 (:action overcome__anxiety__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_anxiety_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anxiety_entertainment) (not (craves_anxiety_onion))))
 (:action succumb__jealousy2__curiosity__ham
  :parameters ()
  :precondition (and (craves_curiosity_ham) (fears_jealousy2_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_jealousy2_curiosity)) (craves_jealousy2_ham)))
 (:action overcome__sciatica__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_sciatica_cherry) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_cherry))))
 (:action overcome__sciatica__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_sciatica_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_muffin))))
 (:action overcome__sciatica__curiosity__ham
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_ham) (craves_sciatica_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_ham))))
 (:action overcome__sciatica__curiosity__beef
  :parameters ()
  :precondition (and (craves_sciatica_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_beef))))
 (:action overcome__sciatica__curiosity__onion
  :parameters ()
  :precondition (and (craves_sciatica_onion) (craves_curiosity_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_onion))))
 (:action overcome__sciatica__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_sciatica_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_flounder))))
 (:action overcome__sciatica__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_sciatica_flounder) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_sciatica_stimulation) (not (craves_sciatica_flounder))))
 (:action overcome__sciatica__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_sciatica_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_sciatica_stimulation) (not (craves_sciatica_cherry))))
 (:action overcome__sciatica__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_sciatica_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_sciatica_stimulation) (not (craves_sciatica_muffin))))
 (:action overcome__sciatica__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_ham) (craves_sciatica_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_sciatica_stimulation) (not (craves_sciatica_ham))))
 (:action overcome__sciatica__stimulation__beef
  :parameters ()
  :precondition (and (craves_sciatica_beef) (not (v4_4)) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_sciatica_stimulation) (not (craves_sciatica_beef))))
 (:action overcome__sciatica__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_sciatica_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_sciatica_stimulation) (not (craves_sciatica_onion))))
 (:action succumb__grief__curiosity__ham
  :parameters ()
  :precondition (and (fears_grief_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_grief_ham) (not (fears_grief_curiosity))))
 (:action overcome__anxiety__satiety__beef
  :parameters ()
  :precondition (and (craves_anxiety_beef) (craves_satiety_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_anxiety_beef)) (fears_anxiety_satiety)))
 (:action overcome__anxiety__satiety__onion
  :parameters ()
  :precondition (and (craves_anxiety_onion) (craves_satiety_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_anxiety_onion)) (fears_anxiety_satiety)))
 (:action overcome__anxiety__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v3_4)) (craves_anxiety_muffin) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_anxiety_muffin)) (fears_anxiety_satiety)))
 (:action overcome__anxiety__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (craves_anxiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_anxiety_ham)) (fears_anxiety_satiety)))
 (:action succumb__grief__stimulation__flounder
  :parameters ()
  :precondition (and (fears_grief_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_grief_stimulation)) (craves_grief_flounder)))
 (:action succumb__grief__stimulation__tuna
  :parameters ()
  :precondition (and (fears_grief_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_grief_stimulation)) (craves_grief_tuna)))
 (:action succumb__grief__stimulation__muffin
  :parameters ()
  :precondition (and (fears_grief_stimulation) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_grief_stimulation)) (craves_grief_muffin)))
 (:action succumb__grief__stimulation__cherry
  :parameters ()
  :precondition (and (fears_grief_stimulation) (craves_stimulation_cherry) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_grief_cherry) (not (fears_grief_stimulation))))
 (:action succumb__grief__stimulation__ham
  :parameters ()
  :precondition (and (fears_grief_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_grief_ham) (not (fears_grief_stimulation))))
 (:action overcome__jealousy2__entertainment__tuna
  :parameters ()
  :precondition (and (craves_jealousy2_tuna) (craves_entertainment_tuna) (not (v0_4)) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy2_entertainment) (not (craves_jealousy2_tuna))))
 (:action overcome__jealousy2__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_jealousy2_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy2_entertainment) (not (craves_jealousy2_cherry))))
 (:action overcome__jealousy2__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_jealousy2_muffin) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy2_entertainment) (not (craves_jealousy2_muffin))))
 (:action overcome__jealousy2__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_jealousy2_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_jealousy2_ham)) (fears_jealousy2_entertainment)))
 (:action succumb__grief__stimulation__onion
  :parameters ()
  :precondition (and (fears_grief_stimulation) (craves_stimulation_onion) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_grief_stimulation)) (craves_grief_onion)))
 (:action succumb__grief__stimulation__beef
  :parameters ()
  :precondition (and (fears_grief_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_grief_stimulation)) (craves_grief_beef)))
 (:action overcome__anxiety__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_anxiety_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_anxiety_muffin)) (fears_anxiety_curiosity)))
 (:action overcome__anxiety__curiosity__ham
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_ham) (craves_anxiety_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_anxiety_curiosity) (not (craves_anxiety_ham))))
 (:action overcome__anxiety__curiosity__beef
  :parameters ()
  :precondition (and (craves_anxiety_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_anxiety_curiosity) (not (craves_anxiety_beef))))
 (:action overcome__anxiety__curiosity__onion
  :parameters ()
  :precondition (and (craves_anxiety_onion) (craves_curiosity_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_anxiety_curiosity) (not (craves_anxiety_onion))))
 (:action succumb__loneliness__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_loneliness_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_loneliness_flounder) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_loneliness_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_loneliness_tuna) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__muffin
  :parameters ()
  :precondition (and (fears_loneliness_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_loneliness_muffin) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_loneliness_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_loneliness_cherry) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_loneliness_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_loneliness_ham) (not (fears_loneliness_entertainment))))
 (:action overcome__anxiety__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_anxiety_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_anxiety_onion)) (fears_anxiety_stimulation)))
 (:action overcome__anxiety__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_anxiety_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_anxiety_muffin)) (fears_anxiety_stimulation)))
 (:action overcome__anxiety__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_ham) (craves_anxiety_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_anxiety_ham)) (fears_anxiety_stimulation)))
 (:action succumb__loneliness__entertainment__onion
  :parameters ()
  :precondition (and (fears_loneliness_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_loneliness_onion) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__beef
  :parameters ()
  :precondition (and (fears_loneliness_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_loneliness_beef) (not (fears_loneliness_entertainment))))
 (:action overcome__anxiety__stimulation__beef
  :parameters ()
  :precondition (and (craves_anxiety_beef) (not (v4_4)) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_anxiety_stimulation) (not (craves_anxiety_beef))))
 (:action overcome__jealousy2__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (craves_jealousy2_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_jealousy2_ham)) (fears_jealousy2_satiety)))
 (:action succumb__depression__curiosity__ham
  :parameters ()
  :precondition (and (craves_curiosity_ham) (fears_depression_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression_ham) (not (fears_depression_curiosity))))
 (:action overcome__jealousy2__satiety__tuna
  :parameters ()
  :precondition (and (craves_jealousy2_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_jealousy2_satiety) (not (craves_jealousy2_tuna))))
 (:action overcome__jealousy2__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (craves_jealousy2_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_jealousy2_satiety) (not (craves_jealousy2_muffin))))
 (:action succumb__prostatitis3__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_prostatitis3_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_prostatitis3_entertainment)) (craves_prostatitis3_cherry)))
 (:action succumb__depression__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_depression_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression_muffin) (not (fears_depression_stimulation))))
 (:action succumb__depression__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_depression_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression_cherry) (not (fears_depression_stimulation))))
 (:action succumb__depression__stimulation__ham
  :parameters ()
  :precondition (and (fears_depression_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression_ham) (not (fears_depression_stimulation))))
 (:action overcome__grief7__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_grief7_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_grief7_beef)) (fears_grief7_entertainment)))
 (:action overcome__grief7__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_onion) (craves_grief7_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief7_entertainment) (not (craves_grief7_onion))))
 (:action overcome__grief7__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (craves_grief7_flounder) (not (v0_4)) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief7_entertainment) (not (craves_grief7_flounder))))
 (:action succumb__depression__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_depression_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression_onion) (not (fears_depression_stimulation))))
 (:action succumb__depression__stimulation__beef
  :parameters ()
  :precondition (and (fears_depression_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_depression_stimulation)) (craves_depression_beef)))
 (:action succumb__depression__stimulation__flounder
  :parameters ()
  :precondition (and (fears_depression_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_depression_stimulation)) (craves_depression_flounder)))
 (:action succumb__depression__stimulation__tuna
  :parameters ()
  :precondition (and (fears_depression_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_depression_stimulation)) (craves_depression_tuna)))
 (:action succumb__prostatitis3__curiosity__muffin
  :parameters ()
  :precondition (and (fears_prostatitis3_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_prostatitis3_curiosity)) (craves_prostatitis3_muffin)))
 (:action succumb__prostatitis3__curiosity__cherry
  :parameters ()
  :precondition (and (fears_prostatitis3_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_prostatitis3_curiosity)) (craves_prostatitis3_cherry)))
 (:action succumb__prostatitis3__curiosity__ham
  :parameters ()
  :precondition (and (fears_prostatitis3_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_prostatitis3_ham) (not (fears_prostatitis3_curiosity))))
 (:action succumb__prostatitis3__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (fears_prostatitis3_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_prostatitis3_onion) (not (fears_prostatitis3_curiosity))))
 (:action succumb__prostatitis3__curiosity__beef
  :parameters ()
  :precondition (and (fears_prostatitis3_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_prostatitis3_curiosity)) (craves_prostatitis3_beef)))
 (:action succumb__prostatitis3__curiosity__flounder
  :parameters ()
  :precondition (and (fears_prostatitis3_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_prostatitis3_flounder) (not (fears_prostatitis3_curiosity))))
 (:action succumb__prostatitis3__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_prostatitis3_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_prostatitis3_tuna) (not (fears_prostatitis3_curiosity))))
 (:action overcome__jealousy2__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (craves_jealousy2_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy2_tuna)) (fears_jealousy2_curiosity)))
 (:action overcome__jealousy2__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_jealousy2_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy2_muffin)) (fears_jealousy2_curiosity)))
 (:action overcome__jealousy2__curiosity__ham
  :parameters ()
  :precondition (and (not (v5_4)) (craves_jealousy2_ham) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy2_ham)) (fears_jealousy2_curiosity)))
 (:action succumb__prostatitis3__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_prostatitis3_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_prostatitis3_cherry) (not (fears_prostatitis3_stimulation))))
 (:action succumb__prostatitis3__stimulation__flounder
  :parameters ()
  :precondition (and (fears_prostatitis3_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_prostatitis3_flounder) (not (fears_prostatitis3_stimulation))))
 (:action succumb__prostatitis3__stimulation__ham
  :parameters ()
  :precondition (and (fears_prostatitis3_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_prostatitis3_ham) (not (fears_prostatitis3_stimulation))))
 (:action succumb__prostatitis3__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_prostatitis3_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_prostatitis3_stimulation)) (craves_prostatitis3_muffin)))
 (:action succumb__prostatitis3__stimulation__beef
  :parameters ()
  :precondition (and (fears_prostatitis3_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_prostatitis3_beef) (not (fears_prostatitis3_stimulation))))
 (:action succumb__prostatitis3__stimulation__tuna
  :parameters ()
  :precondition (and (fears_prostatitis3_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_prostatitis3_tuna) (not (fears_prostatitis3_stimulation))))
 (:action succumb__prostatitis3__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_prostatitis3_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_prostatitis3_onion) (not (fears_prostatitis3_stimulation))))
 (:action succumb__grief__entertainment__muffin
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_grief_entertainment)) (craves_grief_muffin)))
 (:action succumb__grief__entertainment__cherry
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_grief_cherry) (not (fears_grief_entertainment))))
 (:action succumb__grief__entertainment__ham
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_grief_ham) (not (fears_grief_entertainment))))
 (:action overcome__jealousy2__stimulation__tuna
  :parameters ()
  :precondition (and (craves_jealousy2_tuna) (not (v4_4)) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_jealousy2_stimulation) (not (craves_jealousy2_tuna))))
 (:action succumb__grief__entertainment__onion
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_grief_entertainment)) (craves_grief_onion)))
 (:action succumb__grief__entertainment__beef
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_grief_entertainment)) (craves_grief_beef)))
 (:action overcome__jealousy2__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_jealousy2_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_jealousy2_stimulation) (not (craves_jealousy2_muffin))))
 (:action succumb__grief__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_grief_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_grief_flounder) (not (fears_grief_entertainment))))
 (:action overcome__jealousy2__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_ham) (craves_jealousy2_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_jealousy2_ham)) (fears_jealousy2_stimulation)))
 (:action succumb__grief__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_grief_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_grief_tuna) (not (fears_grief_entertainment))))
 (:action succumb__prostatitis3__satiety__flounder
  :parameters ()
  :precondition (and (fears_prostatitis3_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis3_satiety)) (craves_prostatitis3_flounder)))
 (:action succumb__prostatitis3__satiety__tuna
  :parameters ()
  :precondition (and (fears_prostatitis3_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis3_satiety)) (craves_prostatitis3_tuna)))
 (:action succumb__prostatitis3__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_prostatitis3_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis3_satiety)) (craves_prostatitis3_muffin)))
 (:action succumb__prostatitis3__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_prostatitis3_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis3_satiety)) (craves_prostatitis3_cherry)))
 (:action succumb__prostatitis3__satiety__ham
  :parameters ()
  :precondition (and (fears_prostatitis3_satiety) (craves_satiety_ham) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis3_satiety)) (craves_prostatitis3_ham)))
 (:action succumb__prostatitis3__satiety__onion
  :parameters ()
  :precondition (and (fears_prostatitis3_satiety) (craves_satiety_onion) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis3_satiety)) (craves_prostatitis3_onion)))
 (:action succumb__prostatitis3__satiety__beef
  :parameters ()
  :precondition (and (fears_prostatitis3_satiety) (craves_satiety_beef) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_prostatitis3_satiety)) (craves_prostatitis3_beef)))
 (:action overcome__grief7__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (craves_grief7_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_grief7_beef)) (fears_grief7_satiety)))
 (:action overcome__grief7__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v3_4)) (craves_grief7_onion) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_grief7_onion)) (fears_grief7_satiety)))
 (:action overcome__grief7__satiety__flounder
  :parameters ()
  :precondition (and (craves_grief7_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_grief7_flounder)) (fears_grief7_satiety)))
 (:action overcome__depression__curiosity__muffin
  :parameters ()
  :precondition (and (craves_depression_muffin) (not (v5_4)) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_depression_muffin)) (fears_depression_curiosity)))
 (:action overcome__depression__curiosity__ham
  :parameters ()
  :precondition (and (craves_depression_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_depression_ham)) (fears_depression_curiosity)))
 (:action overcome__depression__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_depression_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_depression_onion)) (fears_depression_curiosity)))
 (:action overcome__depression__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_depression_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_depression_curiosity) (not (craves_depression_tuna))))
 (:action overcome__depression__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_flounder) (craves_depression_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_depression_flounder)) (fears_depression_curiosity)))
 (:action overcome__depression__curiosity__cherry
  :parameters ()
  :precondition (and (craves_depression_cherry) (not (v5_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_depression_cherry)) (fears_depression_curiosity)))
 (:action succumb__boils__entertainment__cherry
  :parameters ()
  :precondition (and (fears_boils_entertainment) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_boils_cherry) (not (fears_boils_entertainment))))
 (:action overcome__prostatitis3__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_prostatitis3_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis3_entertainment) (not (craves_prostatitis3_beef))))
 (:action overcome__prostatitis3__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_onion) (craves_prostatitis3_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_prostatitis3_onion)) (fears_prostatitis3_entertainment)))
 (:action overcome__prostatitis3__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_prostatitis3_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis3_entertainment) (not (craves_prostatitis3_flounder))))
 (:action overcome__depression__stimulation__cherry
  :parameters ()
  :precondition (and (craves_depression_cherry) (craves_stimulation_cherry) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_depression_stimulation) (not (craves_depression_cherry))))
 (:action overcome__depression__stimulation__muffin
  :parameters ()
  :precondition (and (craves_depression_muffin) (not (v4_4)) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_depression_muffin)) (fears_depression_stimulation)))
 (:action overcome__depression__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_depression_ham) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_depression_ham)) (fears_depression_stimulation)))
 (:action succumb__boils__curiosity__ham
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils_ham) (not (fears_boils_curiosity))))
 (:action overcome__depression__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_depression_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_depression_stimulation) (not (craves_depression_onion))))
 (:action overcome__depression__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_depression_tuna) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_depression_stimulation) (not (craves_depression_tuna))))
 (:action overcome__depression__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_depression_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_depression_stimulation) (not (craves_depression_flounder))))
 (:action overcome__grief7__curiosity__ham
  :parameters ()
  :precondition (and (craves_grief7_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_grief7_curiosity) (not (craves_grief7_ham))))
 (:action succumb__boils__curiosity__onion
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils_onion) (not (fears_boils_curiosity))))
 (:action succumb__boils__curiosity__beef
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils_beef) (not (fears_boils_curiosity))))
 (:action succumb__boils__curiosity__flounder
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils_flounder) (not (fears_boils_curiosity))))
 (:action succumb__boils__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_boils_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils_tuna) (not (fears_boils_curiosity))))
 (:action succumb__boils__curiosity__muffin
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_boils_curiosity)) (craves_boils_muffin)))
 (:action succumb__boils__curiosity__cherry
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils_cherry) (not (fears_boils_curiosity))))
 (:action overcome__grief7__curiosity__beef
  :parameters ()
  :precondition (and (not (v5_4)) (craves_grief7_beef) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_grief7_beef)) (fears_grief7_curiosity)))
 (:action overcome__grief7__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_grief7_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_grief7_onion)) (fears_grief7_curiosity)))
 (:action overcome__grief7__curiosity__flounder
  :parameters ()
  :precondition (and (craves_grief7_flounder) (not (v5_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_grief7_flounder)) (fears_grief7_curiosity)))
 (:action overcome__grief7__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_beef) (craves_grief7_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_grief7_beef)) (fears_grief7_stimulation)))
 (:action overcome__grief7__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_grief7_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief7_stimulation) (not (craves_grief7_onion))))
 (:action overcome__grief7__stimulation__flounder
  :parameters ()
  :precondition (and (craves_grief7_flounder) (not (v4_4)) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief7_stimulation) (not (craves_grief7_flounder))))
 (:action overcome__depression__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_depression_ham) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression_ham)) (fears_depression_entertainment)))
 (:action succumb__boils__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_boils_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_boils_satiety)) (craves_boils_muffin)))
 (:action succumb__boils__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_boils_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils_cherry) (not (fears_boils_satiety))))
 (:action overcome__depression__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_onion) (craves_depression_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression_onion)) (fears_depression_entertainment)))
 (:action succumb__boils__satiety__ham
  :parameters ()
  :precondition (and (fears_boils_satiety) (craves_satiety_ham) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils_ham) (not (fears_boils_satiety))))
 (:action overcome__depression__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_depression_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_depression_entertainment) (not (craves_depression_tuna))))
 (:action overcome__depression__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_depression_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_depression_entertainment) (not (craves_depression_flounder))))
 (:action overcome__depression__entertainment__cherry
  :parameters ()
  :precondition (and (craves_depression_cherry) (not (v0_4)) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression_cherry)) (fears_depression_entertainment)))
 (:action overcome__depression__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_depression_muffin) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression_muffin)) (fears_depression_entertainment)))
 (:action overcome__prostatitis3__satiety__flounder
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_flounder) (craves_prostatitis3_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_prostatitis3_flounder)) (fears_prostatitis3_satiety)))
 (:action overcome__prostatitis3__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (craves_prostatitis3_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_prostatitis3_satiety) (not (craves_prostatitis3_beef))))
 (:action overcome__prostatitis3__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v3_4)) (craves_prostatitis3_onion) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_prostatitis3_onion)) (fears_prostatitis3_satiety)))
 (:action overcome__grief__curiosity__flounder
  :parameters ()
  :precondition (and (craves_grief_flounder) (not (v5_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_grief_curiosity) (not (craves_grief_flounder))))
 (:action overcome__grief__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_grief_cherry) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_grief_curiosity) (not (craves_grief_cherry))))
 (:action overcome__grief__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_grief_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_grief_curiosity) (not (craves_grief_muffin))))
 (:action overcome__grief__curiosity__ham
  :parameters ()
  :precondition (and (craves_grief_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_grief_curiosity) (not (craves_grief_ham))))
 (:action overcome__grief__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_grief_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_grief_curiosity) (not (craves_grief_onion))))
 (:action overcome__grief__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (craves_grief_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_grief_curiosity) (not (craves_grief_tuna))))
 (:action succumb__dread__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread_entertainment)) (craves_dread_tuna)))
 (:action succumb__dread__entertainment__onion
  :parameters ()
  :precondition (and (fears_dread_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread_entertainment)) (craves_dread_onion)))
 (:action succumb__dread__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread_entertainment)) (craves_dread_cherry)))
 (:action succumb__dread__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread_entertainment)) (craves_dread_flounder)))
 (:action succumb__dread__entertainment__ham
  :parameters ()
  :precondition (and (fears_dread_entertainment) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread_entertainment)) (craves_dread_ham)))
 (:action succumb__dread__entertainment__muffin
  :parameters ()
  :precondition (and (fears_dread_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_dread_muffin) (not (fears_dread_entertainment))))
 (:action succumb__dread__entertainment__beef
  :parameters ()
  :precondition (and (fears_dread_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread_entertainment)) (craves_dread_beef)))
 (:action overcome__grief__stimulation__tuna
  :parameters ()
  :precondition (and (craves_grief_tuna) (not (v4_4)) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief_stimulation) (not (craves_grief_tuna))))
 (:action overcome__grief__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_grief_flounder) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief_stimulation) (not (craves_grief_flounder))))
 (:action overcome__grief__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_grief_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief_stimulation) (not (craves_grief_cherry))))
 (:action overcome__grief__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_grief_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief_stimulation) (not (craves_grief_muffin))))
 (:action overcome__grief__stimulation__ham
  :parameters ()
  :precondition (and (craves_grief_ham) (not (v4_4)) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_grief_ham)) (fears_grief_stimulation)))
 (:action overcome__grief__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_grief_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief_stimulation) (not (craves_grief_onion))))
 (:action overcome__prostatitis3__curiosity__ham
  :parameters ()
  :precondition (and (craves_prostatitis3_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_prostatitis3_ham)) (fears_prostatitis3_curiosity)))
 (:action succumb__dread__curiosity__ham
  :parameters ()
  :precondition (and (fears_dread_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_dread_ham) (not (fears_dread_curiosity))))
 (:action overcome__prostatitis3__curiosity__beef
  :parameters ()
  :precondition (and (craves_prostatitis3_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_prostatitis3_curiosity) (not (craves_prostatitis3_beef))))
 (:action overcome__prostatitis3__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_prostatitis3_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_prostatitis3_onion)) (fears_prostatitis3_curiosity)))
 (:action overcome__prostatitis3__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_flounder) (craves_prostatitis3_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_prostatitis3_flounder)) (fears_prostatitis3_curiosity)))
 (:action overcome__depression__satiety__flounder
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_flounder) (craves_depression_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression_flounder)) (fears_depression_satiety)))
 (:action overcome__depression__satiety__cherry
  :parameters ()
  :precondition (and (craves_depression_cherry) (craves_satiety_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression_cherry)) (fears_depression_satiety)))
 (:action overcome__depression__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (craves_depression_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression_muffin)) (fears_depression_satiety)))
 (:action overcome__depression__satiety__ham
  :parameters ()
  :precondition (and (craves_depression_ham) (craves_satiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression_ham)) (fears_depression_satiety)))
 (:action overcome__depression__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v3_4)) (craves_depression_onion) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression_onion)) (fears_depression_satiety)))
 (:action overcome__depression__satiety__tuna
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_tuna) (craves_depression_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_depression_tuna)) (fears_depression_satiety)))
 (:action overcome__prostatitis3__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_prostatitis3_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_prostatitis3_flounder)) (fears_prostatitis3_stimulation)))
 (:action succumb__dread__stimulation__beef
  :parameters ()
  :precondition (and (fears_dread_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_dread_stimulation)) (craves_dread_beef)))
 (:action succumb__dread__stimulation__tuna
  :parameters ()
  :precondition (and (fears_dread_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_dread_stimulation)) (craves_dread_tuna)))
 (:action succumb__dread__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_dread_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_dread_stimulation)) (craves_dread_onion)))
 (:action feast__entertainment__cherry__beef
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (craves_entertainment_beef) (not (craves_entertainment_cherry))))
 (:action succumb__dread__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_dread_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_dread_stimulation)) (craves_dread_cherry)))
 (:action succumb__dread__stimulation__flounder
  :parameters ()
  :precondition (and (fears_dread_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_dread_stimulation)) (craves_dread_flounder)))
 (:action succumb__dread__stimulation__ham
  :parameters ()
  :precondition (and (fears_dread_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_dread_stimulation)) (craves_dread_ham)))
 (:action succumb__dread__stimulation__muffin
  :parameters ()
  :precondition (and (fears_dread_stimulation) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_dread_muffin) (not (fears_dread_stimulation))))
 (:action feast__entertainment__cherry__flounder
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (craves_entertainment_flounder) (not (craves_entertainment_cherry))))
 (:action feast__entertainment__cherry__onion
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (not (v2_4)) (not (of)))
  :effect (and (when (z_19_0) (v2_0)) (when (not (z_19_0)) (not (v2_0))) (when (z_19_1) (v2_1)) (when (not (z_19_1)) (not (v2_1))) (when (z_19_2) (v2_2)) (when (not (z_19_2)) (not (v2_2))) (when (z_19_3) (v2_3)) (when (not (z_19_3)) (not (v2_3))) (when (z_19_4) (v2_4)) (when (not (z_19_4)) (not (v2_4))) (when (and (v2_4) (not (z_19_4))) (of)) (craves_entertainment_onion) (not (craves_entertainment_cherry))))
 (:action overcome__prostatitis3__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_prostatitis3_beef) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_prostatitis3_stimulation) (not (craves_prostatitis3_beef))))
 (:action overcome__prostatitis3__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_prostatitis3_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_prostatitis3_onion)) (fears_prostatitis3_stimulation)))
 (:action overcome__grief__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_grief_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_grief_cherry)) (fears_grief_entertainment)))
 (:action overcome__grief__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_muffin) (craves_grief_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief_entertainment) (not (craves_grief_muffin))))
 (:action overcome__grief__entertainment__ham
  :parameters ()
  :precondition (and (craves_grief_ham) (not (v0_4)) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_grief_ham)) (fears_grief_entertainment)))
 (:action overcome__grief__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_onion) (craves_grief_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief_entertainment) (not (craves_grief_onion))))
 (:action overcome__grief__entertainment__tuna
  :parameters ()
  :precondition (and (craves_grief_tuna) (craves_entertainment_tuna) (not (v0_4)) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief_entertainment) (not (craves_grief_tuna))))
 (:action overcome__grief__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_grief_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_grief_flounder)) (fears_grief_entertainment)))
 (:action succumb__sciatica__satiety__onion
  :parameters ()
  :precondition (and (fears_sciatica_satiety) (craves_satiety_onion) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_sciatica_satiety)) (craves_sciatica_onion)))
 (:action succumb__sciatica__satiety__beef
  :parameters ()
  :precondition (and (fears_sciatica_satiety) (craves_satiety_beef) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_sciatica_satiety)) (craves_sciatica_beef)))
 (:action overcome__loneliness__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_loneliness_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_tuna))))
 (:action overcome__loneliness__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_loneliness_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_flounder))))
 (:action overcome__loneliness__curiosity__ham
  :parameters ()
  :precondition (and (craves_loneliness_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_ham))))
 (:action succumb__anxiety__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_anxiety_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_anxiety_cherry) (not (fears_anxiety_entertainment))))
 (:action succumb__anxiety__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_anxiety_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anxiety_entertainment)) (craves_anxiety_flounder)))
 (:action succumb__anxiety__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_anxiety_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_anxiety_ham) (not (fears_anxiety_entertainment))))
 (:action succumb__anxiety__entertainment__muffin
  :parameters ()
  :precondition (and (craves_entertainment_muffin) (fears_anxiety_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_anxiety_muffin) (not (fears_anxiety_entertainment))))
 (:action succumb__anxiety__entertainment__beef
  :parameters ()
  :precondition (and (craves_entertainment_beef) (fears_anxiety_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anxiety_entertainment)) (craves_anxiety_beef)))
 (:action succumb__anxiety__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_anxiety_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anxiety_entertainment)) (craves_anxiety_tuna)))
 (:action succumb__anxiety__entertainment__onion
  :parameters ()
  :precondition (and (craves_entertainment_onion) (fears_anxiety_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anxiety_entertainment)) (craves_anxiety_onion)))
 (:action overcome__loneliness__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_loneliness_tuna) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_loneliness_stimulation) (not (craves_loneliness_tuna))))
 (:action overcome__loneliness__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_loneliness_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_loneliness_stimulation) (not (craves_loneliness_flounder))))
 (:action overcome__loneliness__stimulation__ham
  :parameters ()
  :precondition (and (craves_loneliness_ham) (not (v4_4)) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_loneliness_stimulation) (not (craves_loneliness_ham))))
 (:action succumb__anxiety__curiosity__ham
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_anxiety_ham) (not (fears_anxiety_curiosity))))
 (:action overcome__grief__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (not (v3_4)) (craves_grief_onion) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_grief_satiety) (not (craves_grief_onion))))
 (:action overcome__grief__satiety__tuna
  :parameters ()
  :precondition (and (craves_grief_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_grief_satiety) (not (craves_grief_tuna))))
 (:action overcome__grief__satiety__flounder
  :parameters ()
  :precondition (and (craves_grief_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_grief_flounder)) (fears_grief_satiety)))
 (:action overcome__grief__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_grief_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_grief_cherry)) (fears_grief_satiety)))
 (:action overcome__grief__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v3_4)) (craves_grief_muffin) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_grief_satiety) (not (craves_grief_muffin))))
 (:action overcome__grief__satiety__ham
  :parameters ()
  :precondition (and (craves_grief_ham) (craves_satiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_grief_ham)) (fears_grief_satiety)))
 (:action overcome__loneliness__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_loneliness_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_loneliness_tuna)) (fears_loneliness_entertainment)))
 (:action overcome__loneliness__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_loneliness_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_loneliness_flounder)) (fears_loneliness_entertainment)))
 (:action overcome__loneliness__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_loneliness_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_loneliness_cherry)) (fears_loneliness_entertainment)))
 (:action overcome__loneliness__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_loneliness_ham) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_loneliness_ham)) (fears_loneliness_entertainment)))
 (:action succumb__anxiety__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_anxiety_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anxiety_satiety)) (craves_anxiety_onion)))
 (:action succumb__anxiety__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_anxiety_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anxiety_satiety)) (craves_anxiety_beef)))
 (:action succumb__anxiety__satiety__flounder
  :parameters ()
  :precondition (and (craves_satiety_flounder) (fears_anxiety_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anxiety_satiety)) (craves_anxiety_flounder)))
 (:action succumb__anxiety__satiety__tuna
  :parameters ()
  :precondition (and (craves_satiety_tuna) (fears_anxiety_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anxiety_satiety)) (craves_anxiety_tuna)))
 (:action succumb__anxiety__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_anxiety_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anxiety_satiety)) (craves_anxiety_muffin)))
 (:action succumb__anxiety__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_anxiety_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anxiety_satiety)) (craves_anxiety_cherry)))
 (:action succumb__anxiety__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_anxiety_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_anxiety_satiety)) (craves_anxiety_ham)))
 (:action succumb__boils__satiety__onion
  :parameters ()
  :precondition (and (fears_boils_satiety) (craves_satiety_onion) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils_onion) (not (fears_boils_satiety))))
 (:action succumb__boils__satiety__beef
  :parameters ()
  :precondition (and (fears_boils_satiety) (craves_satiety_beef) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils_beef) (not (fears_boils_satiety))))
 (:action succumb__boils__satiety__flounder
  :parameters ()
  :precondition (and (fears_boils_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils_flounder) (not (fears_boils_satiety))))
 (:action succumb__boils__satiety__tuna
  :parameters ()
  :precondition (and (fears_boils_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils_tuna) (not (fears_boils_satiety))))
 (:action overcome__angina__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (not (v5_4)) (craves_angina_tuna) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_angina_curiosity) (not (craves_angina_tuna))))
 (:action overcome__angina__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_angina_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_angina_curiosity) (not (craves_angina_flounder))))
 (:action overcome__angina__curiosity__ham
  :parameters ()
  :precondition (and (craves_angina_ham) (not (v5_4)) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_angina_curiosity) (not (craves_angina_ham))))
 (:action succumb__sciatica__entertainment__cherry
  :parameters ()
  :precondition (and (fears_sciatica_entertainment) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_sciatica_cherry) (not (fears_sciatica_entertainment))))
 (:action overcome__angina__stimulation__ham
  :parameters ()
  :precondition (and (craves_angina_ham) (not (v4_4)) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_angina_stimulation) (not (craves_angina_ham))))
 (:action overcome__angina__stimulation__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (craves_angina_tuna) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_angina_stimulation) (not (craves_angina_tuna))))
 (:action overcome__angina__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_angina_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_angina_stimulation) (not (craves_angina_flounder))))
 (:action succumb__sciatica__curiosity__muffin
  :parameters ()
  :precondition (and (fears_sciatica_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_muffin)))
 (:action succumb__sciatica__curiosity__cherry
  :parameters ()
  :precondition (and (fears_sciatica_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_sciatica_cherry) (not (fears_sciatica_curiosity))))
 (:action succumb__sciatica__curiosity__ham
  :parameters ()
  :precondition (and (fears_sciatica_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_ham)))
 (:action succumb__sciatica__curiosity__onion
  :parameters ()
  :precondition (and (fears_sciatica_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_onion)))
 (:action succumb__sciatica__curiosity__beef
  :parameters ()
  :precondition (and (fears_sciatica_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_sciatica_beef) (not (fears_sciatica_curiosity))))
 (:action succumb__sciatica__curiosity__flounder
  :parameters ()
  :precondition (and (fears_sciatica_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_flounder)))
 (:action succumb__sciatica__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_sciatica_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_tuna)))
 (:action overcome__loneliness__satiety__tuna
  :parameters ()
  :precondition (and (not (v3_4)) (craves_loneliness_tuna) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_loneliness_tuna)) (fears_loneliness_satiety)))
 (:action overcome__loneliness__satiety__flounder
  :parameters ()
  :precondition (and (not (v3_4)) (craves_satiety_flounder) (craves_loneliness_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_loneliness_flounder)) (fears_loneliness_satiety)))
 (:action overcome__loneliness__satiety__ham
  :parameters ()
  :precondition (and (craves_loneliness_ham) (craves_satiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_loneliness_ham)) (fears_loneliness_satiety)))
 (:action overcome__angina__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_angina_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_angina_entertainment) (not (craves_angina_tuna))))
 (:action overcome__angina__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_angina_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_angina_entertainment) (not (craves_angina_flounder))))
 (:action overcome__angina__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_angina_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_angina_entertainment) (not (craves_angina_cherry))))
 (:action overcome__angina__entertainment__ham
  :parameters ()
  :precondition (and (craves_angina_ham) (not (v0_4)) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_angina_entertainment) (not (craves_angina_ham))))
 (:action succumb__sciatica__satiety__flounder
  :parameters ()
  :precondition (and (fears_sciatica_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_sciatica_satiety)) (craves_sciatica_flounder)))
 (:action succumb__sciatica__satiety__tuna
  :parameters ()
  :precondition (and (fears_sciatica_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_sciatica_satiety)) (craves_sciatica_tuna)))
 (:action succumb__sciatica__satiety__muffin
  :parameters ()
  :precondition (and (fears_sciatica_satiety) (craves_satiety_muffin) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_sciatica_satiety)) (craves_sciatica_muffin)))
 (:action succumb__sciatica__satiety__cherry
  :parameters ()
  :precondition (and (fears_sciatica_satiety) (craves_satiety_cherry) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_sciatica_satiety)) (craves_sciatica_cherry)))
 (:action succumb__sciatica__satiety__ham
  :parameters ()
  :precondition (and (fears_sciatica_satiety) (craves_satiety_ham) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_sciatica_satiety)) (craves_sciatica_ham)))
 (:action overcome__laceration__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_laceration_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_laceration_curiosity) (not (craves_laceration_muffin))))
 (:action overcome__laceration__curiosity__ham
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_ham) (craves_laceration_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_laceration_curiosity) (not (craves_laceration_ham))))
 (:action succumb__depression1__stimulation__ham
  :parameters ()
  :precondition (and (craves_stimulation_ham) (fears_depression1_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression1_ham) (not (fears_depression1_stimulation))))
 (:action overcome__laceration__curiosity__beef
  :parameters ()
  :precondition (and (craves_laceration_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_laceration_beef)) (fears_laceration_curiosity)))
 (:action overcome__laceration__curiosity__onion
  :parameters ()
  :precondition (and (craves_laceration_onion) (craves_curiosity_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_laceration_curiosity) (not (craves_laceration_onion))))
 (:action overcome__laceration__curiosity__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (craves_laceration_flounder) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_laceration_flounder)) (fears_laceration_curiosity)))
 (:action overcome__laceration__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_laceration_cherry) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_laceration_cherry)) (fears_laceration_curiosity)))
 (:action succumb__depression1__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_depression1_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression1_onion) (not (fears_depression1_stimulation))))
 (:action succumb__depression1__stimulation__beef
  :parameters ()
  :precondition (and (craves_stimulation_beef) (fears_depression1_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression1_beef) (not (fears_depression1_stimulation))))
 (:action succumb__depression1__stimulation__flounder
  :parameters ()
  :precondition (and (craves_stimulation_flounder) (fears_depression1_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression1_flounder) (not (fears_depression1_stimulation))))
 (:action succumb__depression1__stimulation__tuna
  :parameters ()
  :precondition (and (fears_depression1_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression1_tuna) (not (fears_depression1_stimulation))))
 (:action succumb__depression1__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_depression1_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression1_muffin) (not (fears_depression1_stimulation))))
 (:action succumb__depression1__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_depression1_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_depression1_cherry) (not (fears_depression1_stimulation))))
 (:action overcome__laceration__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_laceration_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_laceration_cherry)) (fears_laceration_stimulation)))
 (:action succumb__anger__entertainment__beef
  :parameters ()
  :precondition (and (fears_anger_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_beef)))
 (:action overcome__laceration__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_laceration_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_laceration_stimulation) (not (craves_laceration_muffin))))
 (:action overcome__laceration__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_ham) (craves_laceration_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_laceration_stimulation) (not (craves_laceration_ham))))
 (:action succumb__anger__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_anger_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_tuna)))
 (:action succumb__anger__entertainment__onion
  :parameters ()
  :precondition (and (craves_entertainment_onion) (fears_anger_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_onion)))
 (:action succumb__anger__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_anger_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_cherry)))
 (:action succumb__anger__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_anger_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_flounder)))
 (:action succumb__anger__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_anger_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_ham)))
 (:action succumb__anger__entertainment__muffin
  :parameters ()
  :precondition (and (fears_anger_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_muffin)))
 (:action overcome__laceration__stimulation__beef
  :parameters ()
  :precondition (and (craves_laceration_beef) (not (v4_4)) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_laceration_beef)) (fears_laceration_stimulation)))
 (:action overcome__laceration__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_laceration_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_laceration_stimulation) (not (craves_laceration_onion))))
 (:action overcome__laceration__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_flounder) (craves_laceration_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_laceration_flounder)) (fears_laceration_stimulation)))
 (:action succumb__anger__curiosity__ham
  :parameters ()
  :precondition (and (craves_curiosity_ham) (fears_anger_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_anger_curiosity)) (craves_anger_ham)))
 (:action overcome__angina__satiety__ham
  :parameters ()
  :precondition (and (craves_angina_ham) (craves_satiety_ham) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_angina_satiety) (not (craves_angina_ham))))
 (:action overcome__angina__satiety__tuna
  :parameters ()
  :precondition (and (craves_angina_tuna) (not (v3_4)) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_angina_satiety) (not (craves_angina_tuna))))
 (:action overcome__angina__satiety__flounder
  :parameters ()
  :precondition (and (not (v3_4)) (craves_angina_flounder) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_angina_satiety) (not (craves_angina_flounder))))
 (:action succumb__dread8__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_dread8_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_dread8_entertainment)) (craves_dread8_cherry)))
 (:action succumb__anger__stimulation__ham
  :parameters ()
  :precondition (and (craves_stimulation_ham) (fears_anger_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_anger_stimulation)) (craves_anger_ham)))
 (:action succumb__anger__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_anger_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_anger_stimulation)) (craves_anger_muffin)))
 (:action overcome__laceration__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_laceration_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_laceration_entertainment) (not (craves_laceration_ham))))
 (:action succumb__anger__stimulation__beef
  :parameters ()
  :precondition (and (craves_stimulation_beef) (fears_anger_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_anger_stimulation)) (craves_anger_beef)))
 (:action succumb__anger__stimulation__tuna
  :parameters ()
  :precondition (and (fears_anger_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_anger_tuna) (not (fears_anger_stimulation))))
 (:action succumb__anger__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_anger_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_anger_stimulation)) (craves_anger_onion)))
 (:action succumb__anger__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_anger_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_anger_stimulation)) (craves_anger_cherry)))
 (:action succumb__anger__stimulation__flounder
  :parameters ()
  :precondition (and (fears_anger_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_anger_stimulation)) (craves_anger_flounder)))
 (:action succumb__dread8__curiosity__ham
  :parameters ()
  :precondition (and (craves_curiosity_ham) (fears_dread8_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_dread8_curiosity)) (craves_dread8_ham)))
 (:action succumb__dread8__curiosity__muffin
  :parameters ()
  :precondition (and (craves_curiosity_muffin) (fears_dread8_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_dread8_muffin) (not (fears_dread8_curiosity))))
 (:action overcome__laceration__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_laceration_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_laceration_beef)) (fears_laceration_entertainment)))
 (:action overcome__laceration__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_laceration_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_laceration_entertainment) (not (craves_laceration_onion))))
 (:action overcome__laceration__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_laceration_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_laceration_entertainment) (not (craves_laceration_flounder))))
 (:action overcome__laceration__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_laceration_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_laceration_cherry)) (fears_laceration_entertainment)))
 (:action overcome__laceration__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_muffin) (craves_laceration_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_laceration_entertainment) (not (craves_laceration_muffin))))
 (:action succumb__dread8__curiosity__beef
  :parameters ()
  :precondition (and (craves_curiosity_beef) (fears_dread8_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_dread8_curiosity)) (craves_dread8_beef)))
 (:action succumb__dread8__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_dread8_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_dread8_curiosity)) (craves_dread8_tuna)))
 (:action succumb__dread8__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (fears_dread8_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_dread8_curiosity)) (craves_dread8_onion)))
 (:action succumb__dread8__curiosity__cherry
  :parameters ()
  :precondition (and (fears_dread8_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_dread8_curiosity)) (craves_dread8_cherry)))
 (:action succumb__dread8__curiosity__flounder
  :parameters ()
  :precondition (and (fears_dread8_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_dread8_curiosity)) (craves_dread8_flounder)))
 (:action overcome__hangover__curiosity__flounder
  :parameters ()
  :precondition (and (craves_hangover_flounder) (not (v5_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_hangover_curiosity) (not (craves_hangover_flounder))))
 (:action overcome__hangover__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_hangover_cherry) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_hangover_curiosity) (not (craves_hangover_cherry))))
 (:action overcome__hangover__curiosity__muffin
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_muffin) (craves_hangover_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_hangover_curiosity) (not (craves_hangover_muffin))))
 (:action overcome__hangover__curiosity__ham
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_ham) (craves_hangover_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_hangover_curiosity) (not (craves_hangover_ham))))
 (:action overcome__hangover__curiosity__beef
  :parameters ()
  :precondition (and (craves_hangover_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_hangover_curiosity) (not (craves_hangover_beef))))
 (:action overcome__hangover__curiosity__onion
  :parameters ()
  :precondition (and (craves_hangover_onion) (craves_curiosity_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_hangover_curiosity) (not (craves_hangover_onion))))
 (:action overcome__hangover__stimulation__flounder
  :parameters ()
  :precondition (and (not (v4_4)) (craves_hangover_flounder) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_hangover_stimulation) (not (craves_hangover_flounder))))
 (:action overcome__hangover__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_hangover_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_hangover_cherry)) (fears_hangover_stimulation)))
 (:action overcome__hangover__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_hangover_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_hangover_stimulation) (not (craves_hangover_muffin))))
 (:action overcome__hangover__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_ham) (craves_hangover_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_hangover_stimulation) (not (craves_hangover_ham))))
 (:action overcome__hangover__stimulation__beef
  :parameters ()
  :precondition (and (craves_hangover_beef) (not (v4_4)) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_hangover_stimulation) (not (craves_hangover_beef))))
 (:action overcome__hangover__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (craves_hangover_onion) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_hangover_stimulation) (not (craves_hangover_onion))))
 (:action succumb__jealousy__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_jealousy_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_jealousy_satiety)) (craves_jealousy_beef)))
 (:action succumb__jealousy__satiety__tuna
  :parameters ()
  :precondition (and (craves_satiety_tuna) (fears_jealousy_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_jealousy_tuna) (not (fears_jealousy_satiety))))
 (:action succumb__jealousy__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_jealousy_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_jealousy_onion) (not (fears_jealousy_satiety))))
 (:action succumb__jealousy__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_jealousy_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_jealousy_satiety)) (craves_jealousy_cherry)))
 (:action succumb__jealousy__satiety__flounder
  :parameters ()
  :precondition (and (craves_satiety_flounder) (fears_jealousy_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_jealousy_flounder) (not (fears_jealousy_satiety))))
 (:action succumb__jealousy__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_jealousy_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_jealousy_satiety)) (craves_jealousy_ham)))
 (:action succumb__jealousy__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_jealousy_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_jealousy_satiety)) (craves_jealousy_muffin)))
 (:action overcome__laceration__satiety__flounder
  :parameters ()
  :precondition (and (craves_laceration_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_laceration_satiety) (not (craves_laceration_flounder))))
 (:action overcome__laceration__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v3_4)) (craves_laceration_cherry) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_laceration_satiety) (not (craves_laceration_cherry))))
 (:action overcome__laceration__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v3_4)) (craves_laceration_muffin) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_laceration_satiety) (not (craves_laceration_muffin))))
 (:action overcome__laceration__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (not (v3_4)) (craves_laceration_ham) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_laceration_satiety) (not (craves_laceration_ham))))
 (:action overcome__laceration__satiety__beef
  :parameters ()
  :precondition (and (craves_laceration_beef) (craves_satiety_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_laceration_satiety) (not (craves_laceration_beef))))
 (:action overcome__laceration__satiety__onion
  :parameters ()
  :precondition (and (craves_laceration_onion) (craves_satiety_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_laceration_satiety) (not (craves_laceration_onion))))
 (:action succumb__depression1__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_depression1_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_depression1_entertainment)) (craves_depression1_cherry)))
 (:action overcome__hangover__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_hangover_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_hangover_cherry)) (fears_hangover_entertainment)))
 (:action overcome__hangover__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_hangover_muffin) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_hangover_entertainment) (not (craves_hangover_muffin))))
 (:action overcome__hangover__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_hangover_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_hangover_ham)) (fears_hangover_entertainment)))
 (:action overcome__hangover__entertainment__beef
  :parameters ()
  :precondition (and (craves_hangover_beef) (not (v0_4)) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_hangover_entertainment) (not (craves_hangover_beef))))
 (:action overcome__hangover__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_hangover_onion) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_hangover_entertainment) (not (craves_hangover_onion))))
 (:action overcome__hangover__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_hangover_flounder) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_hangover_entertainment) (not (craves_hangover_flounder))))
 (:action succumb__depression1__curiosity__beef
  :parameters ()
  :precondition (and (fears_depression1_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression1_beef) (not (fears_depression1_curiosity))))
 (:action succumb__depression1__curiosity__tuna
  :parameters ()
  :precondition (and (fears_depression1_curiosity) (craves_curiosity_tuna) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_depression1_curiosity)) (craves_depression1_tuna)))
 (:action succumb__depression1__curiosity__onion
  :parameters ()
  :precondition (and (fears_depression1_curiosity) (craves_curiosity_onion) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_depression1_curiosity)) (craves_depression1_onion)))
 (:action succumb__depression1__curiosity__cherry
  :parameters ()
  :precondition (and (fears_depression1_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_depression1_curiosity)) (craves_depression1_cherry)))
 (:action succumb__depression1__curiosity__flounder
  :parameters ()
  :precondition (and (fears_depression1_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression1_flounder) (not (fears_depression1_curiosity))))
 (:action succumb__depression1__curiosity__ham
  :parameters ()
  :precondition (and (fears_depression1_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_depression1_curiosity)) (craves_depression1_ham)))
 (:action succumb__depression1__curiosity__muffin
  :parameters ()
  :precondition (and (fears_depression1_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_depression1_muffin) (not (fears_depression1_curiosity))))
 (:action overcome__jealousy__curiosity__onion
  :parameters ()
  :precondition (and (craves_jealousy_onion) (craves_curiosity_onion) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy_onion)) (fears_jealousy_curiosity)))
 (:action overcome__jealousy__curiosity__muffin
  :parameters ()
  :precondition (and (craves_jealousy_muffin) (not (v5_4)) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy_muffin)) (fears_jealousy_curiosity)))
 (:action overcome__jealousy__curiosity__ham
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_ham) (craves_jealousy_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_jealousy_curiosity) (not (craves_jealousy_ham))))
 (:action succumb__boils4__stimulation__flounder
  :parameters ()
  :precondition (and (fears_boils4_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_boils4_flounder) (not (fears_boils4_stimulation))))
 (:action succumb__boils4__stimulation__tuna
  :parameters ()
  :precondition (and (fears_boils4_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_boils4_tuna) (not (fears_boils4_stimulation))))
 (:action succumb__boils4__stimulation__muffin
  :parameters ()
  :precondition (and (fears_boils4_stimulation) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_boils4_muffin) (not (fears_boils4_stimulation))))
 (:action succumb__boils4__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_boils4_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_boils4_stimulation)) (craves_boils4_cherry)))
 (:action succumb__boils4__stimulation__ham
  :parameters ()
  :precondition (and (fears_boils4_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_boils4_ham) (not (fears_boils4_stimulation))))
 (:action overcome__jealousy__curiosity__beef
  :parameters ()
  :precondition (and (craves_jealousy_beef) (not (v5_4)) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_jealousy_beef)) (fears_jealousy_curiosity)))
 (:action succumb__boils4__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_boils4_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_boils4_onion) (not (fears_boils4_stimulation))))
 (:action succumb__boils4__stimulation__beef
  :parameters ()
  :precondition (and (fears_boils4_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_boils4_beef) (not (fears_boils4_stimulation))))
 (:action succumb__abrasion__entertainment__cherry
  :parameters ()
  :precondition (and (fears_abrasion_entertainment) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_cherry)))
 (:action succumb__abrasion__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_abrasion_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_flounder)))
 (:action succumb__abrasion__entertainment__ham
  :parameters ()
  :precondition (and (fears_abrasion_entertainment) (craves_entertainment_ham) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_ham)))
 (:action succumb__abrasion__entertainment__muffin
  :parameters ()
  :precondition (and (fears_abrasion_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_abrasion_muffin) (not (fears_abrasion_entertainment))))
 (:action overcome__jealousy__stimulation__beef
  :parameters ()
  :precondition (and (craves_jealousy_beef) (not (v4_4)) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_jealousy_beef)) (fears_jealousy_stimulation)))
 (:action overcome__jealousy__stimulation__onion
  :parameters ()
  :precondition (and (craves_jealousy_onion) (craves_stimulation_onion) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_jealousy_onion)) (fears_jealousy_stimulation)))
 (:action succumb__abrasion__entertainment__beef
  :parameters ()
  :precondition (and (fears_abrasion_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_beef)))
 (:action overcome__jealousy__stimulation__muffin
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_muffin) (craves_jealousy_muffin) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_jealousy_muffin)) (fears_jealousy_stimulation)))
 (:action overcome__jealousy__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_ham) (craves_jealousy_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_jealousy_stimulation) (not (craves_jealousy_ham))))
 (:action succumb__abrasion__entertainment__tuna
  :parameters ()
  :precondition (and (fears_abrasion_entertainment) (craves_entertainment_tuna) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_tuna)))
 (:action succumb__abrasion__entertainment__onion
  :parameters ()
  :precondition (and (fears_abrasion_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_abrasion_onion) (not (fears_abrasion_entertainment))))
 (:action succumb__boils4__satiety__tuna
  :parameters ()
  :precondition (and (fears_boils4_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils4_tuna) (not (fears_boils4_satiety))))
 (:action succumb__boils4__satiety__onion
  :parameters ()
  :precondition (and (fears_boils4_satiety) (craves_satiety_onion) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils4_onion) (not (fears_boils4_satiety))))
 (:action succumb__boils4__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_boils4_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_boils4_satiety)) (craves_boils4_cherry)))
 (:action succumb__boils4__satiety__flounder
  :parameters ()
  :precondition (and (fears_boils4_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils4_flounder) (not (fears_boils4_satiety))))
 (:action succumb__boils4__satiety__ham
  :parameters ()
  :precondition (and (fears_boils4_satiety) (craves_satiety_ham) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils4_ham) (not (fears_boils4_satiety))))
 (:action succumb__boils4__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_boils4_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_boils4_muffin) (not (fears_boils4_satiety))))
 (:action succumb__boils4__satiety__beef
  :parameters ()
  :precondition (and (fears_boils4_satiety) (craves_satiety_beef) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_boils4_satiety)) (craves_boils4_beef)))
 (:action succumb__abrasion__curiosity__ham
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_ham)))
 (:action overcome__hangover__satiety__onion
  :parameters ()
  :precondition (and (craves_hangover_onion) (craves_satiety_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_hangover_onion)) (fears_hangover_satiety)))
 (:action overcome__hangover__satiety__flounder
  :parameters ()
  :precondition (and (craves_hangover_flounder) (not (v3_4)) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_hangover_flounder)) (fears_hangover_satiety)))
 (:action overcome__hangover__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_hangover_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_hangover_cherry)) (fears_hangover_satiety)))
 (:action overcome__hangover__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (not (v3_4)) (craves_hangover_muffin) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_hangover_satiety) (not (craves_hangover_muffin))))
 (:action overcome__hangover__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (not (v3_4)) (craves_hangover_ham) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_hangover_ham)) (fears_hangover_satiety)))
 (:action overcome__hangover__satiety__beef
  :parameters ()
  :precondition (and (craves_hangover_beef) (craves_satiety_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_hangover_satiety) (not (craves_hangover_beef))))
 (:action succumb__abrasion__stimulation__tuna
  :parameters ()
  :precondition (and (fears_abrasion_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_abrasion_stimulation)) (craves_abrasion_tuna)))
 (:action succumb__abrasion__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_abrasion_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_abrasion_onion) (not (fears_abrasion_stimulation))))
 (:action succumb__abrasion__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_abrasion_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_abrasion_stimulation)) (craves_abrasion_cherry)))
 (:action succumb__abrasion__stimulation__flounder
  :parameters ()
  :precondition (and (craves_stimulation_flounder) (fears_abrasion_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_abrasion_stimulation)) (craves_abrasion_flounder)))
 (:action succumb__abrasion__stimulation__ham
  :parameters ()
  :precondition (and (craves_stimulation_ham) (fears_abrasion_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_abrasion_stimulation)) (craves_abrasion_ham)))
 (:action succumb__abrasion__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_abrasion_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_abrasion_stimulation)) (craves_abrasion_muffin)))
 (:action overcome__jealousy__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_jealousy_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_jealousy_cherry)) (fears_jealousy_entertainment)))
 (:action overcome__jealousy__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_jealousy_muffin) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_jealousy_muffin)) (fears_jealousy_entertainment)))
 (:action overcome__jealousy__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_jealousy_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy_entertainment) (not (craves_jealousy_ham))))
 (:action succumb__abrasion__stimulation__beef
  :parameters ()
  :precondition (and (craves_stimulation_beef) (fears_abrasion_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_abrasion_stimulation)) (craves_abrasion_beef)))
 (:action overcome__jealousy__entertainment__beef
  :parameters ()
  :precondition (and (craves_jealousy_beef) (not (v0_4)) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_jealousy_beef)) (fears_jealousy_entertainment)))
 (:action overcome__jealousy__entertainment__onion
  :parameters ()
  :precondition (and (craves_jealousy_onion) (not (v0_4)) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_jealousy_entertainment) (not (craves_jealousy_onion))))
 (:action overcome__depression1__curiosity__beef
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_beef) (craves_depression1_beef) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_depression1_curiosity) (not (craves_depression1_beef))))
 (:action overcome__depression1__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (not (v5_4)) (craves_depression1_onion) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_depression1_curiosity) (not (craves_depression1_onion))))
 (:action overcome__depression1__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (craves_depression1_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_depression1_curiosity) (not (craves_depression1_tuna))))
 (:action overcome__depression1__curiosity__flounder
  :parameters ()
  :precondition (and (craves_depression1_flounder) (not (v5_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_depression1_curiosity) (not (craves_depression1_flounder))))
 (:action overcome__depression1__curiosity__cherry
  :parameters ()
  :precondition (and (craves_depression1_cherry) (not (v5_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_depression1_cherry)) (fears_depression1_curiosity)))
 (:action overcome__depression1__curiosity__ham
  :parameters ()
  :precondition (and (not (v5_4)) (craves_depression1_ham) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_depression1_curiosity) (not (craves_depression1_ham))))
 (:action succumb__dread8__stimulation__muffin
  :parameters ()
  :precondition (and (craves_stimulation_muffin) (fears_dread8_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_dread8_muffin) (not (fears_dread8_stimulation))))
 (:action succumb__dread8__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_dread8_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_dread8_stimulation)) (craves_dread8_cherry)))
 (:action succumb__dread8__stimulation__ham
  :parameters ()
  :precondition (and (craves_stimulation_ham) (fears_dread8_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_dread8_ham) (not (fears_dread8_stimulation))))
 (:action succumb__dread8__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_dread8_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_dread8_onion) (not (fears_dread8_stimulation))))
 (:action succumb__dread8__stimulation__beef
  :parameters ()
  :precondition (and (craves_stimulation_beef) (fears_dread8_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_dread8_beef) (not (fears_dread8_stimulation))))
 (:action succumb__dread8__stimulation__flounder
  :parameters ()
  :precondition (and (fears_dread8_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_dread8_stimulation)) (craves_dread8_flounder)))
 (:action succumb__dread8__stimulation__tuna
  :parameters ()
  :precondition (and (fears_dread8_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_dread8_stimulation)) (craves_dread8_tuna)))
 (:action overcome__depression1__stimulation__ham
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_ham) (craves_depression1_ham) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_depression1_stimulation) (not (craves_depression1_ham))))
 (:action succumb__prostatitis__entertainment__ham
  :parameters ()
  :precondition (and (craves_entertainment_ham) (fears_prostatitis_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_prostatitis_ham) (not (fears_prostatitis_entertainment))))
 (:action succumb__prostatitis__entertainment__muffin
  :parameters ()
  :precondition (and (fears_prostatitis_entertainment) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_prostatitis_entertainment)) (craves_prostatitis_muffin)))
 (:action overcome__depression1__stimulation__beef
  :parameters ()
  :precondition (and (not (v4_4)) (craves_stimulation_beef) (craves_depression1_beef) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_depression1_beef)) (fears_depression1_stimulation)))
 (:action succumb__prostatitis__entertainment__beef
  :parameters ()
  :precondition (and (fears_prostatitis_entertainment) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_prostatitis_entertainment)) (craves_prostatitis_beef)))
 (:action overcome__depression1__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (not (v4_4)) (craves_depression1_onion) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_depression1_stimulation) (not (craves_depression1_onion))))
 (:action overcome__depression1__stimulation__tuna
  :parameters ()
  :precondition (and (craves_depression1_tuna) (not (v4_4)) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_depression1_tuna)) (fears_depression1_stimulation)))
 (:action succumb__prostatitis__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (fears_prostatitis_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_prostatitis_entertainment)) (craves_prostatitis_tuna)))
 (:action overcome__depression1__stimulation__flounder
  :parameters ()
  :precondition (and (craves_depression1_flounder) (not (v4_4)) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_depression1_flounder)) (fears_depression1_stimulation)))
 (:action succumb__prostatitis__entertainment__onion
  :parameters ()
  :precondition (and (fears_prostatitis_entertainment) (craves_entertainment_onion) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_prostatitis_entertainment)) (craves_prostatitis_onion)))
 (:action overcome__depression1__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_depression1_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_depression1_cherry)) (fears_depression1_stimulation)))
 (:action succumb__prostatitis__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_prostatitis_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_prostatitis_entertainment)) (craves_prostatitis_cherry)))
 (:action succumb__prostatitis__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_prostatitis_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (craves_prostatitis_flounder) (not (fears_prostatitis_entertainment))))
 (:action succumb__dread8__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (fears_dread8_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread8_cherry) (not (fears_dread8_satiety))))
 (:action succumb__dread8__satiety__flounder
  :parameters ()
  :precondition (and (fears_dread8_satiety) (craves_satiety_flounder) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_dread8_satiety)) (craves_dread8_flounder)))
 (:action succumb__dread8__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (fears_dread8_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread8_ham) (not (fears_dread8_satiety))))
 (:action succumb__dread8__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (fears_dread8_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread8_muffin) (not (fears_dread8_satiety))))
 (:action overcome__anger__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_onion) (craves_anger_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anger_entertainment) (not (craves_anger_onion))))
 (:action overcome__anger__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_anger_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anger_entertainment) (not (craves_anger_tuna))))
 (:action overcome__anger__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (craves_anger_flounder) (not (v0_4)) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_anger_flounder)) (fears_anger_entertainment)))
 (:action succumb__dread8__satiety__beef
  :parameters ()
  :precondition (and (craves_satiety_beef) (fears_dread8_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread8_beef) (not (fears_dread8_satiety))))
 (:action overcome__anger__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_anger_cherry) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anger_entertainment) (not (craves_anger_cherry))))
 (:action overcome__anger__entertainment__muffin
  :parameters ()
  :precondition (and (not (v0_4)) (craves_anger_muffin) (craves_entertainment_muffin) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anger_entertainment) (not (craves_anger_muffin))))
 (:action succumb__dread8__satiety__tuna
  :parameters ()
  :precondition (and (fears_dread8_satiety) (craves_satiety_tuna) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (not (fears_dread8_satiety)) (craves_dread8_tuna)))
 (:action overcome__anger__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_anger_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_anger_entertainment) (not (craves_anger_ham))))
 (:action succumb__dread8__satiety__onion
  :parameters ()
  :precondition (and (craves_satiety_onion) (fears_dread8_satiety) (not (of)))
  :effect (and (when (z_15_0) (v3_0)) (when (not (z_15_0)) (not (v3_0))) (when (z_15_1) (v3_1)) (when (not (z_15_1)) (not (v3_1))) (when (z_15_2) (v3_2)) (when (not (z_15_2)) (not (v3_2))) (when (z_15_3) (v3_3)) (when (not (z_15_3)) (not (v3_3))) (when (z_15_4) (v3_4)) (when (not (z_15_4)) (not (v3_4))) (when (and (not (v3_4)) (z_15_4)) (of)) (craves_dread8_onion) (not (fears_dread8_satiety))))
 (:action succumb__prostatitis__curiosity__ham
  :parameters ()
  :precondition (and (fears_prostatitis_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_prostatitis_ham) (not (fears_prostatitis_curiosity))))
 (:action overcome__jealousy__satiety__beef
  :parameters ()
  :precondition (and (craves_jealousy_beef) (craves_satiety_beef) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_jealousy_beef)) (fears_jealousy_satiety)))
 (:action overcome__jealousy__satiety__onion
  :parameters ()
  :precondition (and (craves_jealousy_onion) (craves_satiety_onion) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_jealousy_satiety) (not (craves_jealousy_onion))))
 (:action overcome__jealousy__satiety__muffin
  :parameters ()
  :precondition (and (craves_satiety_muffin) (craves_jealousy_muffin) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_jealousy_muffin)) (fears_jealousy_satiety)))
 (:action overcome__jealousy__satiety__ham
  :parameters ()
  :precondition (and (craves_satiety_ham) (not (v3_4)) (craves_jealousy_ham) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_jealousy_satiety) (not (craves_jealousy_ham))))
 (:action succumb__boils4__entertainment__cherry
  :parameters ()
  :precondition (and (craves_entertainment_cherry) (fears_boils4_entertainment) (not (of)))
  :effect (and (when (z_11_0) (v0_0)) (when (not (z_11_0)) (not (v0_0))) (when (z_11_1) (v0_1)) (when (not (z_11_1)) (not (v0_1))) (when (z_11_2) (v0_2)) (when (not (z_11_2)) (not (v0_2))) (when (z_11_3) (v0_3)) (when (not (z_11_3)) (not (v0_3))) (when (z_11_4) (v0_4)) (when (not (z_11_4)) (not (v0_4))) (when (and (not (v0_4)) (z_11_4)) (of)) (not (fears_boils4_entertainment)) (craves_boils4_cherry)))
 (:action succumb__prostatitis__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (fears_prostatitis_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_prostatitis_stimulation)) (craves_prostatitis_cherry)))
 (:action succumb__prostatitis__stimulation__flounder
  :parameters ()
  :precondition (and (fears_prostatitis_stimulation) (craves_stimulation_flounder) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_prostatitis_flounder) (not (fears_prostatitis_stimulation))))
 (:action succumb__prostatitis__stimulation__ham
  :parameters ()
  :precondition (and (fears_prostatitis_stimulation) (craves_stimulation_ham) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (craves_prostatitis_ham) (not (fears_prostatitis_stimulation))))
 (:action succumb__prostatitis__stimulation__muffin
  :parameters ()
  :precondition (and (fears_prostatitis_stimulation) (craves_stimulation_muffin) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_prostatitis_stimulation)) (craves_prostatitis_muffin)))
 (:action overcome__depression1__entertainment__beef
  :parameters ()
  :precondition (and (not (v0_4)) (craves_depression1_beef) (craves_entertainment_beef) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression1_beef)) (fears_depression1_entertainment)))
 (:action overcome__depression1__entertainment__onion
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_onion) (craves_depression1_onion) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_depression1_entertainment) (not (craves_depression1_onion))))
 (:action overcome__depression1__entertainment__tuna
  :parameters ()
  :precondition (and (craves_entertainment_tuna) (not (v0_4)) (craves_depression1_tuna) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression1_tuna)) (fears_depression1_entertainment)))
 (:action overcome__depression1__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (craves_depression1_flounder) (not (v0_4)) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression1_flounder)) (fears_depression1_entertainment)))
 (:action overcome__depression1__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_depression1_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_depression1_cherry)) (fears_depression1_entertainment)))
 (:action succumb__prostatitis__stimulation__beef
  :parameters ()
  :precondition (and (fears_prostatitis_stimulation) (craves_stimulation_beef) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_prostatitis_stimulation)) (craves_prostatitis_beef)))
 (:action overcome__depression1__entertainment__ham
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_ham) (craves_depression1_ham) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_depression1_entertainment) (not (craves_depression1_ham))))
 (:action succumb__prostatitis__stimulation__tuna
  :parameters ()
  :precondition (and (fears_prostatitis_stimulation) (craves_stimulation_tuna) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_prostatitis_stimulation)) (craves_prostatitis_tuna)))
 (:action succumb__prostatitis__stimulation__onion
  :parameters ()
  :precondition (and (craves_stimulation_onion) (fears_prostatitis_stimulation) (not (of)))
  :effect (and (when (z_16_0) (v4_0)) (when (not (z_16_0)) (not (v4_0))) (when (z_16_1) (v4_1)) (when (not (z_16_1)) (not (v4_1))) (when (z_16_2) (v4_2)) (when (not (z_16_2)) (not (v4_2))) (when (z_16_3) (v4_3)) (when (not (z_16_3)) (not (v4_3))) (when (z_16_4) (v4_4)) (when (not (z_16_4)) (not (v4_4))) (when (and (not (v4_4)) (z_16_4)) (of)) (not (fears_prostatitis_stimulation)) (craves_prostatitis_onion)))
 (:action succumb__boils4__curiosity__cherry
  :parameters ()
  :precondition (and (fears_boils4_curiosity) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (not (fears_boils4_curiosity)) (craves_boils4_cherry)))
 (:action succumb__boils4__curiosity__flounder
  :parameters ()
  :precondition (and (fears_boils4_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils4_flounder) (not (fears_boils4_curiosity))))
 (:action succumb__boils4__curiosity__ham
  :parameters ()
  :precondition (and (fears_boils4_curiosity) (craves_curiosity_ham) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils4_ham) (not (fears_boils4_curiosity))))
 (:action succumb__boils4__curiosity__muffin
  :parameters ()
  :precondition (and (fears_boils4_curiosity) (craves_curiosity_muffin) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils4_muffin) (not (fears_boils4_curiosity))))
 (:action succumb__boils4__curiosity__beef
  :parameters ()
  :precondition (and (fears_boils4_curiosity) (craves_curiosity_beef) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils4_beef) (not (fears_boils4_curiosity))))
 (:action succumb__boils4__curiosity__tuna
  :parameters ()
  :precondition (and (craves_curiosity_tuna) (fears_boils4_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils4_tuna) (not (fears_boils4_curiosity))))
 (:action succumb__boils4__curiosity__onion
  :parameters ()
  :precondition (and (craves_curiosity_onion) (fears_boils4_curiosity) (not (of)))
  :effect (and (when (z_18_0) (v5_0)) (when (not (z_18_0)) (not (v5_0))) (when (z_18_1) (v5_1)) (when (not (z_18_1)) (not (v5_1))) (when (z_18_2) (v5_2)) (when (not (z_18_2)) (not (v5_2))) (when (z_18_3) (v5_3)) (when (not (z_18_3)) (not (v5_3))) (when (z_18_4) (v5_4)) (when (not (z_18_4)) (not (v5_4))) (when (and (not (v5_4)) (z_18_4)) (of)) (craves_boils4_onion) (not (fears_boils4_curiosity))))
 (:action overcome__dread8__curiosity__cherry
  :parameters ()
  :precondition (and (craves_dread8_cherry) (not (v5_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_dread8_cherry)) (fears_dread8_curiosity)))
 (:action overcome__dread8__stimulation__cherry
  :parameters ()
  :precondition (and (craves_dread8_cherry) (craves_stimulation_cherry) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_dread8_stimulation) (not (craves_dread8_cherry))))
 (:action overcome__dread8__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_dread8_cherry) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_dread8_entertainment) (not (craves_dread8_cherry))))
 (:action overcome__boils4__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_cherry) (craves_boils4_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_boils4_cherry)) (fears_boils4_curiosity)))
 (:action overcome__boils4__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_boils4_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (not (craves_boils4_cherry)) (fears_boils4_stimulation)))
 (:action overcome__dread8__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (craves_dread8_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (fears_dread8_satiety) (not (craves_dread8_cherry))))
 (:action overcome__boils4__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_boils4_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (not (craves_boils4_cherry)) (fears_boils4_entertainment)))
 (:action overcome__boils4__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v3_4)) (craves_boils4_cherry) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_boils4_cherry)) (fears_boils4_satiety)))
 (:action overcome__grief7__entertainment__cherry
  :parameters ()
  :precondition (and (craves_grief7_cherry) (not (v0_4)) (craves_entertainment_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_grief7_entertainment) (not (craves_grief7_cherry))))
 (:action overcome__grief7__satiety__cherry
  :parameters ()
  :precondition (and (craves_grief7_cherry) (craves_satiety_cherry) (not (v3_4)) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_grief7_cherry)) (fears_grief7_satiety)))
 (:action overcome__prostatitis3__entertainment__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_cherry) (craves_prostatitis3_cherry) (not (of)))
  :effect (and (when (z_14_0) (v0_0)) (when (not (z_14_0)) (not (v0_0))) (when (z_14_1) (v0_1)) (when (not (z_14_1)) (not (v0_1))) (when (z_14_2) (v0_2)) (when (not (z_14_2)) (not (v0_2))) (when (z_14_3) (v0_3)) (when (not (z_14_3)) (not (v0_3))) (when (z_14_4) (v0_4)) (when (not (z_14_4)) (not (v0_4))) (when (and (v0_4) (not (z_14_4))) (of)) (fears_prostatitis3_entertainment) (not (craves_prostatitis3_cherry))))
 (:action overcome__grief7__curiosity__cherry
  :parameters ()
  :precondition (and (craves_grief7_cherry) (not (v5_4)) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (not (craves_grief7_cherry)) (fears_grief7_curiosity)))
 (:action overcome__grief7__stimulation__cherry
  :parameters ()
  :precondition (and (craves_grief7_cherry) (craves_stimulation_cherry) (not (v4_4)) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_grief7_stimulation) (not (craves_grief7_cherry))))
 (:action overcome__prostatitis3__satiety__cherry
  :parameters ()
  :precondition (and (craves_satiety_cherry) (not (v3_4)) (craves_prostatitis3_cherry) (not (of)))
  :effect (and (when (z_21_0) (v3_0)) (when (not (z_21_0)) (not (v3_0))) (when (z_21_1) (v3_1)) (when (not (z_21_1)) (not (v3_1))) (when (z_21_2) (v3_2)) (when (not (z_21_2)) (not (v3_2))) (when (z_21_3) (v3_3)) (when (not (z_21_3)) (not (v3_3))) (when (z_21_4) (v3_4)) (when (not (z_21_4)) (not (v3_4))) (when (and (v3_4) (not (z_21_4))) (of)) (not (craves_prostatitis3_cherry)) (fears_prostatitis3_satiety)))
 (:action overcome__prostatitis3__curiosity__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (craves_prostatitis3_cherry) (craves_curiosity_cherry) (not (of)))
  :effect (and (when (z_3_0) (v5_0)) (when (not (z_3_0)) (not (v5_0))) (when (z_3_1) (v5_1)) (when (not (z_3_1)) (not (v5_1))) (when (z_3_2) (v5_2)) (when (not (z_3_2)) (not (v5_2))) (when (z_3_3) (v5_3)) (when (not (z_3_3)) (not (v5_3))) (when (z_3_4) (v5_4)) (when (not (z_3_4)) (not (v5_4))) (when (and (v5_4) (not (z_3_4))) (of)) (fears_prostatitis3_curiosity) (not (craves_prostatitis3_cherry))))
 (:action overcome__prostatitis3__stimulation__cherry
  :parameters ()
  :precondition (and (craves_stimulation_cherry) (not (v4_4)) (craves_prostatitis3_cherry) (not (of)))
  :effect (and (when (z_1_0) (v4_0)) (when (not (z_1_0)) (not (v4_0))) (when (z_1_1) (v4_1)) (when (not (z_1_1)) (not (v4_1))) (when (z_1_2) (v4_2)) (when (not (z_1_2)) (not (v4_2))) (when (z_1_3) (v4_3)) (when (not (z_1_3)) (not (v4_3))) (when (z_1_4) (v4_4)) (when (not (z_1_4)) (not (v4_4))) (when (and (v4_4) (not (z_1_4))) (of)) (fears_prostatitis3_stimulation) (not (craves_prostatitis3_cherry))))
)
