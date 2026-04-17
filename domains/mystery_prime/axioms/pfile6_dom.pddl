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
             (harmony_intoxication_0)
             (harmony_intoxication_1)
             (harmony_intoxication_2)
             (harmony_intoxication_3)
             (harmony_intoxication_4)
             (locale_hotdog_0)
             (locale_hotdog_1)
             (locale_hotdog_2)
             (locale_hotdog_3)
             (locale_hotdog_4)
             (locale_flounder_0)
             (locale_flounder_1)
             (locale_flounder_2)
             (locale_flounder_3)
             (locale_flounder_4)
             (harmony_excitement_0)
             (harmony_excitement_1)
             (harmony_excitement_2)
             (harmony_excitement_3)
             (harmony_excitement_4)
             (v0_0)
             (v0_1)
             (v0_2)
             (v0_3)
             (v0_4)
             (v1_0)
             (v1_1)
             (v1_2)
             (v1_3)
             (v1_4)
             (locale_apple_0)
             (locale_apple_1)
             (locale_apple_2)
             (locale_apple_3)
             (locale_apple_4)
             (v2_0)
             (v2_1)
             (v2_2)
             (v2_3)
             (v2_4)
             (locale_wurst_0)
             (locale_wurst_1)
             (locale_wurst_2)
             (locale_wurst_3)
             (locale_wurst_4)
             (locale_guava_0)
             (locale_guava_1)
             (locale_guava_2)
             (locale_guava_3)
             (locale_guava_4)
             (locale_hamburger_0)
             (locale_hamburger_1)
             (locale_hamburger_2)
             (locale_hamburger_3)
             (locale_hamburger_4)
             (locale_haroset_0)
             (locale_haroset_1)
             (locale_haroset_2)
             (locale_haroset_3)
             (locale_haroset_4)
             (v3_0)
             (v3_1)
             (v3_2)
             (v3_3)
             (v3_4)
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
             (v8_0)
             (v8_1)
             (v8_2)
             (v8_3)
             (v8_4)
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
             (harmony_curiosity_0)
             (harmony_curiosity_1)
             (harmony_curiosity_2)
             (harmony_curiosity_3)
             (harmony_curiosity_4)
             (craves_sciatica_haroset)
             (craves_sciatica_wurst)
             (craves_sciatica_hotdog)
             (craves_sciatica_guava)
             (fears_jealousy_curiosity)
             (fears_boils_entertainment)
             (craves_boils_apple)
             (craves_boils_flounder)
             (craves_boils_haroset)
             (craves_boils_hamburger)
             (craves_boils_wurst)
             (craves_boils_hotdog)
             (fears_abrasion_excitement)
             (fears_sciatica_intoxication)
             (fears_jealousy_excitement)
             (fears_boils_intoxication)
             (fears_sciatica_curiosity)
             (fears_dread_entertainment)
             (craves_dread_apple)
             (craves_dread_haroset)
             (craves_dread_hamburger)
             (craves_dread_wurst)
             (craves_hangover_haroset)
             (craves_dread_hotdog)
             (craves_loneliness_haroset)
             (craves_dread_guava)
             (craves_laceration_wurst)
             (craves_anxiety_wurst)
             (fears_boils_curiosity)
             (fears_laceration_entertainment)
             (craves_anger_haroset)
             (craves_laceration_apple)
             (craves_curiosity_hotdog)
             (craves_laceration_flounder)
             (craves_dread_flounder)
             (craves_intoxication_haroset)
             (craves_laceration_haroset)
             (craves_angina_guava)
             (craves_laceration_hamburger)
             (craves_laceration_hotdog)
             (craves_excitement_guava)
             (craves_laceration_guava)
             (craves_jealousy_guava)
             (craves_sciatica_flounder)
             (fears_sciatica_excitement)
             (craves_grief_guava)
             (fears_dread_intoxication)
             (craves_boils_guava)
             (fears_boils_excitement)
             (fears_laceration_intoxication)
             (fears_dread_curiosity)
             (craves_entertainment_flounder)
             (fears_laceration_curiosity)
             (fears_anxiety_entertainment)
             (craves_prostatitis_haroset)
             (craves_anxiety_apple)
             (craves_abrasion_haroset)
             (craves_anxiety_flounder)
             (craves_sciatica_hamburger)
             (craves_anxiety_haroset)
             (craves_jealousy_wurst)
             (craves_anxiety_hamburger)
             (craves_anxiety_hotdog)
             (craves_anxiety_guava)
             (fears_dread_excitement)
             (fears_laceration_excitement)
             (fears_anxiety_intoxication)
             (fears_anxiety_curiosity)
             (fears_hangover_entertainment)
             (craves_hangover_apple)
             (craves_hangover_flounder)
             (craves_hangover_hamburger)
             (craves_hangover_wurst)
             (craves_hangover_hotdog)
             (craves_hangover_guava)
             (fears_anxiety_excitement)
             (fears_hangover_intoxication)
             (fears_hangover_curiosity)
             (fears_anger_entertainment)
             (craves_anger_apple)
             (craves_entertainment_apple)
             (craves_anger_flounder)
             (craves_entertainment_guava)
             (craves_intoxication_apple)
             (craves_anger_hamburger)
             (craves_intoxication_guava)
             (craves_anger_wurst)
             (craves_curiosity_apple)
             (craves_anger_hotdog)
             (craves_curiosity_guava)
             (craves_excitement_apple)
             (craves_anger_guava)
             (fears_hangover_excitement)
             (fears_anger_intoxication)
             (fears_anger_curiosity)
             (fears_loneliness_entertainment)
             (craves_entertainment_hamburger)
             (fears_grief_entertainment)
             (craves_entertainment_haroset)
             (fears_anger_excitement)
             (fears_loneliness_intoxication)
             (craves_intoxication_hamburger)
             (craves_curiosity_hamburger)
             (craves_curiosity_haroset)
             (craves_excitement_hamburger)
             (craves_excitement_haroset)
             (craves_intoxication_flounder)
             (craves_curiosity_flounder)
             (craves_excitement_flounder)
             (craves_entertainment_hotdog)
             (craves_entertainment_wurst)
             (craves_intoxication_hotdog)
             (craves_intoxication_wurst)
             (craves_curiosity_wurst)
             (craves_excitement_hotdog)
             (craves_excitement_wurst)
             (fears_grief_intoxication)
             (craves_grief_hotdog)
             (craves_grief_wurst)
             (fears_loneliness_curiosity)
             (craves_loneliness_apple)
             (fears_prostatitis_entertainment)
             (craves_prostatitis_apple)
             (craves_loneliness_flounder)
             (craves_prostatitis_flounder)
             (craves_loneliness_hamburger)
             (craves_prostatitis_hamburger)
             (craves_loneliness_wurst)
             (craves_prostatitis_wurst)
             (craves_loneliness_hotdog)
             (craves_prostatitis_hotdog)
             (craves_loneliness_guava)
             (craves_prostatitis_guava)
             (fears_grief_curiosity)
             (craves_grief_apple)
             (fears_angina_entertainment)
             (craves_angina_apple)
             (craves_grief_haroset)
             (craves_angina_flounder)
             (craves_grief_flounder)
             (craves_angina_haroset)
             (craves_angina_hamburger)
             (craves_grief_hamburger)
             (craves_angina_wurst)
             (craves_angina_hotdog)
             (fears_loneliness_excitement)
             (fears_prostatitis_intoxication)
             (fears_grief_excitement)
             (fears_angina_intoxication)
             (fears_prostatitis_curiosity)
             (fears_abrasion_entertainment)
             (craves_abrasion_apple)
             (craves_abrasion_flounder)
             (craves_abrasion_hamburger)
             (craves_abrasion_wurst)
             (craves_abrasion_hotdog)
             (craves_abrasion_guava)
             (fears_angina_curiosity)
             (fears_jealousy_entertainment)
             (craves_jealousy_apple)
             (craves_jealousy_flounder)
             (craves_jealousy_haroset)
             (craves_jealousy_hamburger)
             (craves_jealousy_hotdog)
             (fears_prostatitis_excitement)
             (fears_abrasion_intoxication)
             (fears_angina_excitement)
             (fears_jealousy_intoxication)
             (fears_abrasion_curiosity)
             (fears_sciatica_entertainment)
             (craves_sciatica_apple)
             (of)
 )
 (:derived (c_0_-1) (false))
(:derived (c_0_0) (or (v6_0) (and (c_0_-1) (not (v6_0)))))
(:derived (c_0_1) (and (c_0_0) (v6_1)))
(:derived (c_0_2) (and (c_0_1) (v6_2)))
(:derived (c_0_3) (and (c_0_2) (v6_3)))
(:derived (c_0_4) (and (c_0_3) (v6_4)))
(:derived (z_0_0) (or (and (not (v6_0)) (not (c_0_-1))) (and (c_0_-1) (v6_0))))
(:derived (z_0_1) (or (and (v6_1) (not (c_0_0))) (and (c_0_0) (not (v6_1)))))
(:derived (z_0_2) (or (and (v6_2) (not (c_0_1))) (and (c_0_1) (not (v6_2)))))
(:derived (z_0_3) (or (and (v6_3) (not (c_0_2))) (and (c_0_2) (not (v6_3)))))
(:derived (z_0_4) (or (and (v6_4) (not (c_0_3))) (and (c_0_3) (not (v6_4)))))
(:derived (c_1_-1) (false))
(:derived (c_1_0) (or (v7_0) (and (c_1_-1) (not (v7_0)))))
(:derived (c_1_1) (and (c_1_0) (v7_1)))
(:derived (c_1_2) (and (c_1_1) (v7_2)))
(:derived (c_1_3) (and (c_1_2) (v7_3)))
(:derived (c_1_4) (and (c_1_3) (v7_4)))
(:derived (z_1_0) (or (and (not (v7_0)) (not (c_1_-1))) (and (c_1_-1) (v7_0))))
(:derived (z_1_1) (or (and (v7_1) (not (c_1_0))) (and (c_1_0) (not (v7_1)))))
(:derived (z_1_2) (or (and (v7_2) (not (c_1_1))) (and (c_1_1) (not (v7_2)))))
(:derived (z_1_3) (or (and (v7_3) (not (c_1_2))) (and (c_1_2) (not (v7_3)))))
(:derived (z_1_4) (or (and (v7_4) (not (c_1_3))) (and (c_1_3) (not (v7_4)))))
(:derived (c_2_-1) (false))
(:derived (c_2_0) (or (v0_0) (and (c_2_-1) (not (v0_0)))))
(:derived (c_2_1) (or (v0_1) (and (c_2_0) (not (v0_1)))))
(:derived (c_2_2) (or (v0_2) (and (c_2_1) (not (v0_2)))))
(:derived (c_2_3) (or (v0_3) (and (c_2_2) (not (v0_3)))))
(:derived (c_2_4) (or (v0_4) (and (c_2_3) (not (v0_4)))))
(:derived (z_2_0) (or (and (not (v0_0)) (not (c_2_-1))) (and (c_2_-1) (v0_0))))
(:derived (z_2_1) (or (and (not (v0_1)) (not (c_2_0))) (and (c_2_0) (v0_1))))
(:derived (z_2_2) (or (and (not (v0_2)) (not (c_2_1))) (and (c_2_1) (v0_2))))
(:derived (z_2_3) (or (and (not (v0_3)) (not (c_2_2))) (and (c_2_2) (v0_3))))
(:derived (z_2_4) (or (and (not (v0_4)) (not (c_2_3))) (and (c_2_3) (v0_4))))
(:derived (c_3_-1) (false))
(:derived (c_3_0) (or (v8_0) (and (c_3_-1) (not (v8_0)))))
(:derived (c_3_1) (and (c_3_0) (v8_1)))
(:derived (c_3_2) (and (c_3_1) (v8_2)))
(:derived (c_3_3) (and (c_3_2) (v8_3)))
(:derived (c_3_4) (and (c_3_3) (v8_4)))
(:derived (z_3_0) (or (and (not (v8_0)) (not (c_3_-1))) (and (c_3_-1) (v8_0))))
(:derived (z_3_1) (or (and (v8_1) (not (c_3_0))) (and (c_3_0) (not (v8_1)))))
(:derived (z_3_2) (or (and (v8_2) (not (c_3_1))) (and (c_3_1) (not (v8_2)))))
(:derived (z_3_3) (or (and (v8_3) (not (c_3_2))) (and (c_3_2) (not (v8_3)))))
(:derived (z_3_4) (or (and (v8_4) (not (c_3_3))) (and (c_3_3) (not (v8_4)))))
(:derived (c_4_-1) (false))
(:derived (c_4_0) (or (v1_0) (and (c_4_-1) (not (v1_0)))))
(:derived (c_4_1) (or (v1_1) (and (c_4_0) (not (v1_1)))))
(:derived (c_4_2) (or (v1_2) (and (c_4_1) (not (v1_2)))))
(:derived (c_4_3) (or (v1_3) (and (c_4_2) (not (v1_3)))))
(:derived (c_4_4) (or (v1_4) (and (c_4_3) (not (v1_4)))))
(:derived (z_4_0) (or (and (not (v1_0)) (not (c_4_-1))) (and (c_4_-1) (v1_0))))
(:derived (z_4_1) (or (and (not (v1_1)) (not (c_4_0))) (and (c_4_0) (v1_1))))
(:derived (z_4_2) (or (and (not (v1_2)) (not (c_4_1))) (and (c_4_1) (v1_2))))
(:derived (z_4_3) (or (and (not (v1_3)) (not (c_4_2))) (and (c_4_2) (v1_3))))
(:derived (z_4_4) (or (and (not (v1_4)) (not (c_4_3))) (and (c_4_3) (v1_4))))
(:derived (c_5_-1) (false))
(:derived (c_5_0) (or (v2_0) (and (c_5_-1) (not (v2_0)))))
(:derived (c_5_1) (or (v2_1) (and (c_5_0) (not (v2_1)))))
(:derived (c_5_2) (or (v2_2) (and (c_5_1) (not (v2_2)))))
(:derived (c_5_3) (or (v2_3) (and (c_5_2) (not (v2_3)))))
(:derived (c_5_4) (or (v2_4) (and (c_5_3) (not (v2_4)))))
(:derived (z_5_0) (or (and (not (v2_0)) (not (c_5_-1))) (and (c_5_-1) (v2_0))))
(:derived (z_5_1) (or (and (not (v2_1)) (not (c_5_0))) (and (c_5_0) (v2_1))))
(:derived (z_5_2) (or (and (not (v2_2)) (not (c_5_1))) (and (c_5_1) (v2_2))))
(:derived (z_5_3) (or (and (not (v2_3)) (not (c_5_2))) (and (c_5_2) (v2_3))))
(:derived (z_5_4) (or (and (not (v2_4)) (not (c_5_3))) (and (c_5_3) (v2_4))))
(:derived (c_6_-1) (false))
(:derived (c_6_0) (or (v9_0) (and (c_6_-1) (not (v9_0)))))
(:derived (c_6_1) (and (c_6_0) (v9_1)))
(:derived (c_6_2) (and (c_6_1) (v9_2)))
(:derived (c_6_3) (and (c_6_2) (v9_3)))
(:derived (c_6_4) (and (c_6_3) (v9_4)))
(:derived (z_6_0) (or (and (not (v9_0)) (not (c_6_-1))) (and (c_6_-1) (v9_0))))
(:derived (z_6_1) (or (and (v9_1) (not (c_6_0))) (and (c_6_0) (not (v9_1)))))
(:derived (z_6_2) (or (and (v9_2) (not (c_6_1))) (and (c_6_1) (not (v9_2)))))
(:derived (z_6_3) (or (and (v9_3) (not (c_6_2))) (and (c_6_2) (not (v9_3)))))
(:derived (z_6_4) (or (and (v9_4) (not (c_6_3))) (and (c_6_3) (not (v9_4)))))
(:derived (c_7_-1) (false))
(:derived (c_7_0) (or (v3_0) (and (c_7_-1) (not (v3_0)))))
(:derived (c_7_1) (or (v3_1) (and (c_7_0) (not (v3_1)))))
(:derived (c_7_2) (or (v3_2) (and (c_7_1) (not (v3_2)))))
(:derived (c_7_3) (or (v3_3) (and (c_7_2) (not (v3_3)))))
(:derived (c_7_4) (or (v3_4) (and (c_7_3) (not (v3_4)))))
(:derived (z_7_0) (or (and (not (v3_0)) (not (c_7_-1))) (and (c_7_-1) (v3_0))))
(:derived (z_7_1) (or (and (not (v3_1)) (not (c_7_0))) (and (c_7_0) (v3_1))))
(:derived (z_7_2) (or (and (not (v3_2)) (not (c_7_1))) (and (c_7_1) (v3_2))))
(:derived (z_7_3) (or (and (not (v3_3)) (not (c_7_2))) (and (c_7_2) (v3_3))))
(:derived (z_7_4) (or (and (not (v3_4)) (not (c_7_3))) (and (c_7_3) (v3_4))))
(:derived (c_8_-1) (false))
(:derived (c_8_0) (or (v10_0) (and (c_8_-1) (not (v10_0)))))
(:derived (c_8_1) (and (c_8_0) (v10_1)))
(:derived (c_8_2) (and (c_8_1) (v10_2)))
(:derived (c_8_3) (and (c_8_2) (v10_3)))
(:derived (c_8_4) (and (c_8_3) (v10_4)))
(:derived (z_8_0) (or (and (not (v10_0)) (not (c_8_-1))) (and (c_8_-1) (v10_0))))
(:derived (z_8_1) (or (and (v10_1) (not (c_8_0))) (and (c_8_0) (not (v10_1)))))
(:derived (z_8_2) (or (and (v10_2) (not (c_8_1))) (and (c_8_1) (not (v10_2)))))
(:derived (z_8_3) (or (and (v10_3) (not (c_8_2))) (and (c_8_2) (not (v10_3)))))
(:derived (z_8_4) (or (and (v10_4) (not (c_8_3))) (and (c_8_3) (not (v10_4)))))
(:derived (c_9_-1) (false))
(:derived (c_9_0) (or (v4_0) (and (c_9_-1) (not (v4_0)))))
(:derived (c_9_1) (or (v4_1) (and (c_9_0) (not (v4_1)))))
(:derived (c_9_2) (or (v4_2) (and (c_9_1) (not (v4_2)))))
(:derived (c_9_3) (or (v4_3) (and (c_9_2) (not (v4_3)))))
(:derived (c_9_4) (or (v4_4) (and (c_9_3) (not (v4_4)))))
(:derived (z_9_0) (or (and (not (v4_0)) (not (c_9_-1))) (and (c_9_-1) (v4_0))))
(:derived (z_9_1) (or (and (not (v4_1)) (not (c_9_0))) (and (c_9_0) (v4_1))))
(:derived (z_9_2) (or (and (not (v4_2)) (not (c_9_1))) (and (c_9_1) (v4_2))))
(:derived (z_9_3) (or (and (not (v4_3)) (not (c_9_2))) (and (c_9_2) (v4_3))))
(:derived (z_9_4) (or (and (not (v4_4)) (not (c_9_3))) (and (c_9_3) (v4_4))))
(:derived (c_10_-1) (false))
(:derived (c_10_0) (or (v5_0) (and (c_10_-1) (not (v5_0)))))
(:derived (c_10_1) (or (v5_1) (and (c_10_0) (not (v5_1)))))
(:derived (c_10_2) (or (v5_2) (and (c_10_1) (not (v5_2)))))
(:derived (c_10_3) (or (v5_3) (and (c_10_2) (not (v5_3)))))
(:derived (c_10_4) (or (v5_4) (and (c_10_3) (not (v5_4)))))
(:derived (z_10_0) (or (and (not (v5_0)) (not (c_10_-1))) (and (c_10_-1) (v5_0))))
(:derived (z_10_1) (or (and (not (v5_1)) (not (c_10_0))) (and (c_10_0) (v5_1))))
(:derived (z_10_2) (or (and (not (v5_2)) (not (c_10_1))) (and (c_10_1) (v5_2))))
(:derived (z_10_3) (or (and (not (v5_3)) (not (c_10_2))) (and (c_10_2) (v5_3))))
(:derived (z_10_4) (or (and (not (v5_4)) (not (c_10_3))) (and (c_10_3) (v5_4))))
(:derived (c_11_-1) (false))
(:derived (c_11_0) (or (v6_0) (and (c_11_-1) (not (v6_0)))))
(:derived (c_11_1) (or (v6_1) (and (c_11_0) (not (v6_1)))))
(:derived (c_11_2) (or (v6_2) (and (c_11_1) (not (v6_2)))))
(:derived (c_11_3) (or (v6_3) (and (c_11_2) (not (v6_3)))))
(:derived (c_11_4) (or (v6_4) (and (c_11_3) (not (v6_4)))))
(:derived (z_11_0) (or (and (not (v6_0)) (not (c_11_-1))) (and (c_11_-1) (v6_0))))
(:derived (z_11_1) (or (and (not (v6_1)) (not (c_11_0))) (and (c_11_0) (v6_1))))
(:derived (z_11_2) (or (and (not (v6_2)) (not (c_11_1))) (and (c_11_1) (v6_2))))
(:derived (z_11_3) (or (and (not (v6_3)) (not (c_11_2))) (and (c_11_2) (v6_3))))
(:derived (z_11_4) (or (and (not (v6_4)) (not (c_11_3))) (and (c_11_3) (v6_4))))
(:derived (c_12_-1) (false))
(:derived (c_12_0) (or (v7_0) (and (c_12_-1) (not (v7_0)))))
(:derived (c_12_1) (or (v7_1) (and (c_12_0) (not (v7_1)))))
(:derived (c_12_2) (or (v7_2) (and (c_12_1) (not (v7_2)))))
(:derived (c_12_3) (or (v7_3) (and (c_12_2) (not (v7_3)))))
(:derived (c_12_4) (or (v7_4) (and (c_12_3) (not (v7_4)))))
(:derived (z_12_0) (or (and (not (v7_0)) (not (c_12_-1))) (and (c_12_-1) (v7_0))))
(:derived (z_12_1) (or (and (not (v7_1)) (not (c_12_0))) (and (c_12_0) (v7_1))))
(:derived (z_12_2) (or (and (not (v7_2)) (not (c_12_1))) (and (c_12_1) (v7_2))))
(:derived (z_12_3) (or (and (not (v7_3)) (not (c_12_2))) (and (c_12_2) (v7_3))))
(:derived (z_12_4) (or (and (not (v7_4)) (not (c_12_3))) (and (c_12_3) (v7_4))))
(:derived (c_13_-1) (false))
(:derived (c_13_0) (or (v8_0) (and (c_13_-1) (not (v8_0)))))
(:derived (c_13_1) (or (v8_1) (and (c_13_0) (not (v8_1)))))
(:derived (c_13_2) (or (v8_2) (and (c_13_1) (not (v8_2)))))
(:derived (c_13_3) (or (v8_3) (and (c_13_2) (not (v8_3)))))
(:derived (c_13_4) (or (v8_4) (and (c_13_3) (not (v8_4)))))
(:derived (z_13_0) (or (and (not (v8_0)) (not (c_13_-1))) (and (c_13_-1) (v8_0))))
(:derived (z_13_1) (or (and (not (v8_1)) (not (c_13_0))) (and (c_13_0) (v8_1))))
(:derived (z_13_2) (or (and (not (v8_2)) (not (c_13_1))) (and (c_13_1) (v8_2))))
(:derived (z_13_3) (or (and (not (v8_3)) (not (c_13_2))) (and (c_13_2) (v8_3))))
(:derived (z_13_4) (or (and (not (v8_4)) (not (c_13_3))) (and (c_13_3) (v8_4))))
(:derived (c_14_-1) (false))
(:derived (c_14_0) (or (v9_0) (and (c_14_-1) (not (v9_0)))))
(:derived (c_14_1) (or (v9_1) (and (c_14_0) (not (v9_1)))))
(:derived (c_14_2) (or (v9_2) (and (c_14_1) (not (v9_2)))))
(:derived (c_14_3) (or (v9_3) (and (c_14_2) (not (v9_3)))))
(:derived (c_14_4) (or (v9_4) (and (c_14_3) (not (v9_4)))))
(:derived (z_14_0) (or (and (not (v9_0)) (not (c_14_-1))) (and (c_14_-1) (v9_0))))
(:derived (z_14_1) (or (and (not (v9_1)) (not (c_14_0))) (and (c_14_0) (v9_1))))
(:derived (z_14_2) (or (and (not (v9_2)) (not (c_14_1))) (and (c_14_1) (v9_2))))
(:derived (z_14_3) (or (and (not (v9_3)) (not (c_14_2))) (and (c_14_2) (v9_3))))
(:derived (z_14_4) (or (and (not (v9_4)) (not (c_14_3))) (and (c_14_3) (v9_4))))
(:derived (c_15_-1) (false))
(:derived (c_15_0) (or (v10_0) (and (c_15_-1) (not (v10_0)))))
(:derived (c_15_1) (or (v10_1) (and (c_15_0) (not (v10_1)))))
(:derived (c_15_2) (or (v10_2) (and (c_15_1) (not (v10_2)))))
(:derived (c_15_3) (or (v10_3) (and (c_15_2) (not (v10_3)))))
(:derived (c_15_4) (or (v10_4) (and (c_15_3) (not (v10_4)))))
(:derived (z_15_0) (or (and (not (v10_0)) (not (c_15_-1))) (and (c_15_-1) (v10_0))))
(:derived (z_15_1) (or (and (not (v10_1)) (not (c_15_0))) (and (c_15_0) (v10_1))))
(:derived (z_15_2) (or (and (not (v10_2)) (not (c_15_1))) (and (c_15_1) (v10_2))))
(:derived (z_15_3) (or (and (not (v10_3)) (not (c_15_2))) (and (c_15_2) (v10_3))))
(:derived (z_15_4) (or (and (not (v10_4)) (not (c_15_3))) (and (c_15_3) (v10_4))))
(:derived (c_16_-1) (false))
(:derived (c_16_0) (or (v0_0) (and (c_16_-1) (not (v0_0)))))
(:derived (c_16_1) (and (c_16_0) (v0_1)))
(:derived (c_16_2) (and (c_16_1) (v0_2)))
(:derived (c_16_3) (and (c_16_2) (v0_3)))
(:derived (c_16_4) (and (c_16_3) (v0_4)))
(:derived (z_16_0) (or (and (not (v0_0)) (not (c_16_-1))) (and (c_16_-1) (v0_0))))
(:derived (z_16_1) (or (and (v0_1) (not (c_16_0))) (and (c_16_0) (not (v0_1)))))
(:derived (z_16_2) (or (and (v0_2) (not (c_16_1))) (and (c_16_1) (not (v0_2)))))
(:derived (z_16_3) (or (and (v0_3) (not (c_16_2))) (and (c_16_2) (not (v0_3)))))
(:derived (z_16_4) (or (and (v0_4) (not (c_16_3))) (and (c_16_3) (not (v0_4)))))
(:derived (c_17_-1) (false))
(:derived (c_17_0) (or (v1_0) (and (c_17_-1) (not (v1_0)))))
(:derived (c_17_1) (and (c_17_0) (v1_1)))
(:derived (c_17_2) (and (c_17_1) (v1_2)))
(:derived (c_17_3) (and (c_17_2) (v1_3)))
(:derived (c_17_4) (and (c_17_3) (v1_4)))
(:derived (z_17_0) (or (and (not (v1_0)) (not (c_17_-1))) (and (c_17_-1) (v1_0))))
(:derived (z_17_1) (or (and (v1_1) (not (c_17_0))) (and (c_17_0) (not (v1_1)))))
(:derived (z_17_2) (or (and (v1_2) (not (c_17_1))) (and (c_17_1) (not (v1_2)))))
(:derived (z_17_3) (or (and (v1_3) (not (c_17_2))) (and (c_17_2) (not (v1_3)))))
(:derived (z_17_4) (or (and (v1_4) (not (c_17_3))) (and (c_17_3) (not (v1_4)))))
(:derived (c_18_-1) (false))
(:derived (c_18_0) (or (v2_0) (and (c_18_-1) (not (v2_0)))))
(:derived (c_18_1) (and (c_18_0) (v2_1)))
(:derived (c_18_2) (and (c_18_1) (v2_2)))
(:derived (c_18_3) (and (c_18_2) (v2_3)))
(:derived (c_18_4) (and (c_18_3) (v2_4)))
(:derived (z_18_0) (or (and (not (v2_0)) (not (c_18_-1))) (and (c_18_-1) (v2_0))))
(:derived (z_18_1) (or (and (v2_1) (not (c_18_0))) (and (c_18_0) (not (v2_1)))))
(:derived (z_18_2) (or (and (v2_2) (not (c_18_1))) (and (c_18_1) (not (v2_2)))))
(:derived (z_18_3) (or (and (v2_3) (not (c_18_2))) (and (c_18_2) (not (v2_3)))))
(:derived (z_18_4) (or (and (v2_4) (not (c_18_3))) (and (c_18_3) (not (v2_4)))))
(:derived (c_19_-1) (false))
(:derived (c_19_0) (or (v3_0) (and (c_19_-1) (not (v3_0)))))
(:derived (c_19_1) (and (c_19_0) (v3_1)))
(:derived (c_19_2) (and (c_19_1) (v3_2)))
(:derived (c_19_3) (and (c_19_2) (v3_3)))
(:derived (c_19_4) (and (c_19_3) (v3_4)))
(:derived (z_19_0) (or (and (not (v3_0)) (not (c_19_-1))) (and (c_19_-1) (v3_0))))
(:derived (z_19_1) (or (and (v3_1) (not (c_19_0))) (and (c_19_0) (not (v3_1)))))
(:derived (z_19_2) (or (and (v3_2) (not (c_19_1))) (and (c_19_1) (not (v3_2)))))
(:derived (z_19_3) (or (and (v3_3) (not (c_19_2))) (and (c_19_2) (not (v3_3)))))
(:derived (z_19_4) (or (and (v3_4) (not (c_19_3))) (and (c_19_3) (not (v3_4)))))
(:derived (c_20_-1) (false))
(:derived (c_20_0) (or (v4_0) (and (c_20_-1) (not (v4_0)))))
(:derived (c_20_1) (and (c_20_0) (v4_1)))
(:derived (c_20_2) (and (c_20_1) (v4_2)))
(:derived (c_20_3) (and (c_20_2) (v4_3)))
(:derived (c_20_4) (and (c_20_3) (v4_4)))
(:derived (z_20_0) (or (and (not (v4_0)) (not (c_20_-1))) (and (c_20_-1) (v4_0))))
(:derived (z_20_1) (or (and (v4_1) (not (c_20_0))) (and (c_20_0) (not (v4_1)))))
(:derived (z_20_2) (or (and (v4_2) (not (c_20_1))) (and (c_20_1) (not (v4_2)))))
(:derived (z_20_3) (or (and (v4_3) (not (c_20_2))) (and (c_20_2) (not (v4_3)))))
(:derived (z_20_4) (or (and (v4_4) (not (c_20_3))) (and (c_20_3) (not (v4_4)))))
(:derived (c_21_-1) (false))
(:derived (c_21_0) (or (v5_0) (and (c_21_-1) (not (v5_0)))))
(:derived (c_21_1) (and (c_21_0) (v5_1)))
(:derived (c_21_2) (and (c_21_1) (v5_2)))
(:derived (c_21_3) (and (c_21_2) (v5_3)))
(:derived (c_21_4) (and (c_21_3) (v5_4)))
(:derived (z_21_0) (or (and (not (v5_0)) (not (c_21_-1))) (and (c_21_-1) (v5_0))))
(:derived (z_21_1) (or (and (v5_1) (not (c_21_0))) (and (c_21_0) (not (v5_1)))))
(:derived (z_21_2) (or (and (v5_2) (not (c_21_1))) (and (c_21_1) (not (v5_2)))))
(:derived (z_21_3) (or (and (v5_3) (not (c_21_2))) (and (c_21_2) (not (v5_3)))))
(:derived (z_21_4) (or (and (v5_4) (not (c_21_3))) (and (c_21_3) (not (v5_4)))))
(:action overcome__dread__entertainment__flounder
  :parameters ()
  :precondition (and (craves_dread_flounder) (craves_entertainment_flounder) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_dread_flounder)) (fears_dread_entertainment)))
 (:action overcome__abrasion__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (craves_abrasion_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_abrasion_haroset)) (fears_abrasion_intoxication)))
 (:action feast__curiosity__hotdog__apple
  :parameters ()
  :precondition (and (not (v2_4)) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (craves_curiosity_apple) (not (craves_curiosity_hotdog))))
 (:action feast__entertainment__flounder__wurst
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v3_4)) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (craves_entertainment_wurst) (not (craves_entertainment_flounder))))
 (:action feast__entertainment__flounder__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v3_4)) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (craves_entertainment_hamburger) (not (craves_entertainment_flounder))))
 (:action feast__curiosity__hotdog__wurst
  :parameters ()
  :precondition (and (not (v2_4)) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (craves_curiosity_wurst) (not (craves_curiosity_hotdog))))
 (:action overcome__loneliness__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (craves_loneliness_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_loneliness_haroset)) (fears_loneliness_intoxication)))
 (:action overcome__hangover__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (craves_hangover_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_hangover_haroset)) (fears_hangover_intoxication)))
 (:action overcome__jealousy__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_guava) (craves_jealousy_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_jealousy_excitement) (not (craves_jealousy_guava))))
 (:action overcome__grief__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_guava) (craves_grief_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_grief_guava)) (fears_grief_excitement)))
 (:action drink__apple__hotdog
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (when (z_18_0) (v2_0)) (when (not (z_18_0)) (not (v2_0))) (when (z_18_1) (v2_1)) (when (not (z_18_1)) (not (v2_1))) (when (z_18_2) (v2_2)) (when (not (z_18_2)) (not (v2_2))) (when (z_18_3) (v2_3)) (when (not (z_18_3)) (not (v2_3))) (when (z_18_4) (v2_4)) (when (not (z_18_4)) (not (v2_4))) (when (and (not (v2_4)) (z_18_4)) (of))))
 (:action drink__apple__wurst
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (when (z_3_0) (v8_0)) (when (not (z_3_0)) (not (v8_0))) (when (z_3_1) (v8_1)) (when (not (z_3_1)) (not (v8_1))) (when (z_3_2) (v8_2)) (when (not (z_3_2)) (not (v8_2))) (when (z_3_3) (v8_3)) (when (not (z_3_3)) (not (v8_3))) (when (z_3_4) (v8_4)) (when (not (z_3_4)) (not (v8_4))) (when (and (not (v8_4)) (z_3_4)) (of))))
 (:action drink__apple__guava
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (when (z_1_0) (v7_0)) (when (not (z_1_0)) (not (v7_0))) (when (z_1_1) (v7_1)) (when (not (z_1_1)) (not (v7_1))) (when (z_1_2) (v7_2)) (when (not (z_1_2)) (not (v7_2))) (when (z_1_3) (v7_3)) (when (not (z_1_3)) (not (v7_3))) (when (z_1_4) (v7_4)) (when (not (z_1_4)) (not (v7_4))) (when (and (not (v7_4)) (z_1_4)) (of))))
 (:action drink__apple__flounder
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_19_0) (v3_0)) (when (not (z_19_0)) (not (v3_0))) (when (z_19_1) (v3_1)) (when (not (z_19_1)) (not (v3_1))) (when (z_19_2) (v3_2)) (when (not (z_19_2)) (not (v3_2))) (when (z_19_3) (v3_3)) (when (not (z_19_3)) (not (v3_3))) (when (z_19_4) (v3_4)) (when (not (z_19_4)) (not (v3_4))) (when (and (not (v3_4)) (z_19_4)) (of)) (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of))))
 (:action drink__apple__hamburger
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_6_0) (v9_0)) (when (not (z_6_0)) (not (v9_0))) (when (z_6_1) (v9_1)) (when (not (z_6_1)) (not (v9_1))) (when (z_6_2) (v9_2)) (when (not (z_6_2)) (not (v9_2))) (when (z_6_3) (v9_3)) (when (not (z_6_3)) (not (v9_3))) (when (z_6_4) (v9_4)) (when (not (z_6_4)) (not (v9_4))) (when (and (not (v9_4)) (z_6_4)) (of)) (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of))))
 (:action drink__apple__haroset
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_0_0) (v6_0)) (when (not (z_0_0)) (not (v6_0))) (when (z_0_1) (v6_1)) (when (not (z_0_1)) (not (v6_1))) (when (z_0_2) (v6_2)) (when (not (z_0_2)) (not (v6_2))) (when (z_0_3) (v6_3)) (when (not (z_0_3)) (not (v6_3))) (when (z_0_4) (v6_4)) (when (not (z_0_4)) (not (v6_4))) (when (and (not (v6_4)) (z_0_4)) (of)) (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of))))
 (:action drink__flounder__guava
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (when (z_1_0) (v7_0)) (when (not (z_1_0)) (not (v7_0))) (when (z_1_1) (v7_1)) (when (not (z_1_1)) (not (v7_1))) (when (z_1_2) (v7_2)) (when (not (z_1_2)) (not (v7_2))) (when (z_1_3) (v7_3)) (when (not (z_1_3)) (not (v7_3))) (when (z_1_4) (v7_4)) (when (not (z_1_4)) (not (v7_4))) (when (and (not (v7_4)) (z_1_4)) (of))))
 (:action drink__flounder__hotdog
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_18_0) (v2_0)) (when (not (z_18_0)) (not (v2_0))) (when (z_18_1) (v2_1)) (when (not (z_18_1)) (not (v2_1))) (when (z_18_2) (v2_2)) (when (not (z_18_2)) (not (v2_2))) (when (z_18_3) (v2_3)) (when (not (z_18_3)) (not (v2_3))) (when (z_18_4) (v2_4)) (when (not (z_18_4)) (not (v2_4))) (when (and (not (v2_4)) (z_18_4)) (of)) (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of))))
 (:action drink__flounder__haroset
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_0_0) (v6_0)) (when (not (z_0_0)) (not (v6_0))) (when (z_0_1) (v6_1)) (when (not (z_0_1)) (not (v6_1))) (when (z_0_2) (v6_2)) (when (not (z_0_2)) (not (v6_2))) (when (z_0_3) (v6_3)) (when (not (z_0_3)) (not (v6_3))) (when (z_0_4) (v6_4)) (when (not (z_0_4)) (not (v6_4))) (when (and (not (v6_4)) (z_0_4)) (of)) (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of))))
 (:action drink__flounder__wurst
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (when (z_3_0) (v8_0)) (when (not (z_3_0)) (not (v8_0))) (when (z_3_1) (v8_1)) (when (not (z_3_1)) (not (v8_1))) (when (z_3_2) (v8_2)) (when (not (z_3_2)) (not (v8_2))) (when (z_3_3) (v8_3)) (when (not (z_3_3)) (not (v8_3))) (when (z_3_4) (v8_4)) (when (not (z_3_4)) (not (v8_4))) (when (and (not (v8_4)) (z_3_4)) (of))))
 (:action drink__flounder__hamburger
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_6_0) (v9_0)) (when (not (z_6_0)) (not (v9_0))) (when (z_6_1) (v9_1)) (when (not (z_6_1)) (not (v9_1))) (when (z_6_2) (v9_2)) (when (not (z_6_2)) (not (v9_2))) (when (z_6_3) (v9_3)) (when (not (z_6_3)) (not (v9_3))) (when (z_6_4) (v9_4)) (when (not (z_6_4)) (not (v9_4))) (when (and (not (v9_4)) (z_6_4)) (of)) (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of))))
 (:action drink__flounder__apple
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (when (z_21_0) (v5_0)) (when (not (z_21_0)) (not (v5_0))) (when (z_21_1) (v5_1)) (when (not (z_21_1)) (not (v5_1))) (when (z_21_2) (v5_2)) (when (not (z_21_2)) (not (v5_2))) (when (z_21_3) (v5_3)) (when (not (z_21_3)) (not (v5_3))) (when (z_21_4) (v5_4)) (when (not (z_21_4)) (not (v5_4))) (when (and (not (v5_4)) (z_21_4)) (of))))
 (:action drink__haroset__guava
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (when (z_1_0) (v7_0)) (when (not (z_1_0)) (not (v7_0))) (when (z_1_1) (v7_1)) (when (not (z_1_1)) (not (v7_1))) (when (z_1_2) (v7_2)) (when (not (z_1_2)) (not (v7_2))) (when (z_1_3) (v7_3)) (when (not (z_1_3)) (not (v7_3))) (when (z_1_4) (v7_4)) (when (not (z_1_4)) (not (v7_4))) (when (and (not (v7_4)) (z_1_4)) (of))))
 (:action drink__haroset__hamburger
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_6_0) (v9_0)) (when (not (z_6_0)) (not (v9_0))) (when (z_6_1) (v9_1)) (when (not (z_6_1)) (not (v9_1))) (when (z_6_2) (v9_2)) (when (not (z_6_2)) (not (v9_2))) (when (z_6_3) (v9_3)) (when (not (z_6_3)) (not (v9_3))) (when (z_6_4) (v9_4)) (when (not (z_6_4)) (not (v9_4))) (when (and (not (v9_4)) (z_6_4)) (of)) (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of))))
 (:action drink__haroset__hotdog
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_18_0) (v2_0)) (when (not (z_18_0)) (not (v2_0))) (when (z_18_1) (v2_1)) (when (not (z_18_1)) (not (v2_1))) (when (z_18_2) (v2_2)) (when (not (z_18_2)) (not (v2_2))) (when (z_18_3) (v2_3)) (when (not (z_18_3)) (not (v2_3))) (when (z_18_4) (v2_4)) (when (not (z_18_4)) (not (v2_4))) (when (and (not (v2_4)) (z_18_4)) (of)) (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of))))
 (:action drink__haroset__wurst
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (when (z_3_0) (v8_0)) (when (not (z_3_0)) (not (v8_0))) (when (z_3_1) (v8_1)) (when (not (z_3_1)) (not (v8_1))) (when (z_3_2) (v8_2)) (when (not (z_3_2)) (not (v8_2))) (when (z_3_3) (v8_3)) (when (not (z_3_3)) (not (v8_3))) (when (z_3_4) (v8_4)) (when (not (z_3_4)) (not (v8_4))) (when (and (not (v8_4)) (z_3_4)) (of))))
 (:action drink__haroset__flounder
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_19_0) (v3_0)) (when (not (z_19_0)) (not (v3_0))) (when (z_19_1) (v3_1)) (when (not (z_19_1)) (not (v3_1))) (when (z_19_2) (v3_2)) (when (not (z_19_2)) (not (v3_2))) (when (z_19_3) (v3_3)) (when (not (z_19_3)) (not (v3_3))) (when (z_19_4) (v3_4)) (when (not (z_19_4)) (not (v3_4))) (when (and (not (v3_4)) (z_19_4)) (of)) (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of))))
 (:action drink__haroset__apple
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_21_0) (v5_0)) (when (not (z_21_0)) (not (v5_0))) (when (z_21_1) (v5_1)) (when (not (z_21_1)) (not (v5_1))) (when (z_21_2) (v5_2)) (when (not (z_21_2)) (not (v5_2))) (when (z_21_3) (v5_3)) (when (not (z_21_3)) (not (v5_3))) (when (z_21_4) (v5_4)) (when (not (z_21_4)) (not (v5_4))) (when (and (not (v5_4)) (z_21_4)) (of)) (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of))))
 (:action overcome__prostatitis__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (craves_prostatitis_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_prostatitis_haroset)) (fears_prostatitis_intoxication)))
 (:action overcome__sciatica__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_sciatica_flounder) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_sciatica_flounder)) (fears_sciatica_entertainment)))
 (:action feast__excitement__guava__haroset
  :parameters ()
  :precondition (and (not (v7_4)) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (craves_excitement_haroset) (not (craves_excitement_guava))))
 (:action feast__excitement__guava__apple
  :parameters ()
  :precondition (and (not (v7_4)) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (craves_excitement_apple) (not (craves_excitement_guava))))
 (:action feast__intoxication__haroset__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (not (v6_4)) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (not (craves_intoxication_haroset)) (craves_intoxication_hamburger)))
 (:action feast__intoxication__haroset__guava
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (not (v6_4)) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (not (craves_intoxication_haroset)) (craves_intoxication_guava)))
 (:action overcome__anger__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (not (v1_4)) (craves_anger_haroset) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_anger_haroset)) (fears_anger_intoxication)))
 (:action drink__wurst__hotdog
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (when (z_18_0) (v2_0)) (when (not (z_18_0)) (not (v2_0))) (when (z_18_1) (v2_1)) (when (not (z_18_1)) (not (v2_1))) (when (z_18_2) (v2_2)) (when (not (z_18_2)) (not (v2_2))) (when (z_18_3) (v2_3)) (when (not (z_18_3)) (not (v2_3))) (when (z_18_4) (v2_4)) (when (not (z_18_4)) (not (v2_4))) (when (and (not (v2_4)) (z_18_4)) (of))))
 (:action drink__wurst__guava
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (when (z_1_0) (v7_0)) (when (not (z_1_0)) (not (v7_0))) (when (z_1_1) (v7_1)) (when (not (z_1_1)) (not (v7_1))) (when (z_1_2) (v7_2)) (when (not (z_1_2)) (not (v7_2))) (when (z_1_3) (v7_3)) (when (not (z_1_3)) (not (v7_3))) (when (z_1_4) (v7_4)) (when (not (z_1_4)) (not (v7_4))) (when (and (not (v7_4)) (z_1_4)) (of))))
 (:action drink__wurst__flounder
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_19_0) (v3_0)) (when (not (z_19_0)) (not (v3_0))) (when (z_19_1) (v3_1)) (when (not (z_19_1)) (not (v3_1))) (when (z_19_2) (v3_2)) (when (not (z_19_2)) (not (v3_2))) (when (z_19_3) (v3_3)) (when (not (z_19_3)) (not (v3_3))) (when (z_19_4) (v3_4)) (when (not (z_19_4)) (not (v3_4))) (when (and (not (v3_4)) (z_19_4)) (of)) (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of))))
 (:action drink__wurst__apple
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (when (z_21_0) (v5_0)) (when (not (z_21_0)) (not (v5_0))) (when (z_21_1) (v5_1)) (when (not (z_21_1)) (not (v5_1))) (when (z_21_2) (v5_2)) (when (not (z_21_2)) (not (v5_2))) (when (z_21_3) (v5_3)) (when (not (z_21_3)) (not (v5_3))) (when (z_21_4) (v5_4)) (when (not (z_21_4)) (not (v5_4))) (when (and (not (v5_4)) (z_21_4)) (of))))
 (:action drink__wurst__hamburger
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_6_0) (v9_0)) (when (not (z_6_0)) (not (v9_0))) (when (z_6_1) (v9_1)) (when (not (z_6_1)) (not (v9_1))) (when (z_6_2) (v9_2)) (when (not (z_6_2)) (not (v9_2))) (when (z_6_3) (v9_3)) (when (not (z_6_3)) (not (v9_3))) (when (z_6_4) (v9_4)) (when (not (z_6_4)) (not (v9_4))) (when (and (not (v9_4)) (z_6_4)) (of)) (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of))))
 (:action drink__wurst__haroset
  :parameters ()
  :precondition (and (not (v8_4)) (not (of)))
  :effect (and (when (z_0_0) (v6_0)) (when (not (z_0_0)) (not (v6_0))) (when (z_0_1) (v6_1)) (when (not (z_0_1)) (not (v6_1))) (when (z_0_2) (v6_2)) (when (not (z_0_2)) (not (v6_2))) (when (z_0_3) (v6_3)) (when (not (z_0_3)) (not (v6_3))) (when (z_0_4) (v6_4)) (when (not (z_0_4)) (not (v6_4))) (when (and (not (v6_4)) (z_0_4)) (of)) (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of))))
 (:action drink__hotdog__apple
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (when (z_21_0) (v5_0)) (when (not (z_21_0)) (not (v5_0))) (when (z_21_1) (v5_1)) (when (not (z_21_1)) (not (v5_1))) (when (z_21_2) (v5_2)) (when (not (z_21_2)) (not (v5_2))) (when (z_21_3) (v5_3)) (when (not (z_21_3)) (not (v5_3))) (when (z_21_4) (v5_4)) (when (not (z_21_4)) (not (v5_4))) (when (and (not (v5_4)) (z_21_4)) (of))))
 (:action drink__hotdog__guava
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (when (z_1_0) (v7_0)) (when (not (z_1_0)) (not (v7_0))) (when (z_1_1) (v7_1)) (when (not (z_1_1)) (not (v7_1))) (when (z_1_2) (v7_2)) (when (not (z_1_2)) (not (v7_2))) (when (z_1_3) (v7_3)) (when (not (z_1_3)) (not (v7_3))) (when (z_1_4) (v7_4)) (when (not (z_1_4)) (not (v7_4))) (when (and (not (v7_4)) (z_1_4)) (of))))
 (:action drink__hotdog__haroset
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (when (z_0_0) (v6_0)) (when (not (z_0_0)) (not (v6_0))) (when (z_0_1) (v6_1)) (when (not (z_0_1)) (not (v6_1))) (when (z_0_2) (v6_2)) (when (not (z_0_2)) (not (v6_2))) (when (z_0_3) (v6_3)) (when (not (z_0_3)) (not (v6_3))) (when (z_0_4) (v6_4)) (when (not (z_0_4)) (not (v6_4))) (when (and (not (v6_4)) (z_0_4)) (of))))
 (:action drink__hotdog__flounder
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (when (z_19_0) (v3_0)) (when (not (z_19_0)) (not (v3_0))) (when (z_19_1) (v3_1)) (when (not (z_19_1)) (not (v3_1))) (when (z_19_2) (v3_2)) (when (not (z_19_2)) (not (v3_2))) (when (z_19_3) (v3_3)) (when (not (z_19_3)) (not (v3_3))) (when (z_19_4) (v3_4)) (when (not (z_19_4)) (not (v3_4))) (when (and (not (v3_4)) (z_19_4)) (of))))
 (:action drink__hotdog__wurst
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (when (z_3_0) (v8_0)) (when (not (z_3_0)) (not (v8_0))) (when (z_3_1) (v8_1)) (when (not (z_3_1)) (not (v8_1))) (when (z_3_2) (v8_2)) (when (not (z_3_2)) (not (v8_2))) (when (z_3_3) (v8_3)) (when (not (z_3_3)) (not (v8_3))) (when (z_3_4) (v8_4)) (when (not (z_3_4)) (not (v8_4))) (when (and (not (v8_4)) (z_3_4)) (of))))
 (:action drink__hotdog__hamburger
  :parameters ()
  :precondition (and (not (v2_4)) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (when (z_6_0) (v9_0)) (when (not (z_6_0)) (not (v9_0))) (when (z_6_1) (v9_1)) (when (not (z_6_1)) (not (v9_1))) (when (z_6_2) (v9_2)) (when (not (z_6_2)) (not (v9_2))) (when (z_6_3) (v9_3)) (when (not (z_6_3)) (not (v9_3))) (when (z_6_4) (v9_4)) (when (not (z_6_4)) (not (v9_4))) (when (and (not (v9_4)) (z_6_4)) (of))))
 (:action drink__guava__flounder
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_19_0) (v3_0)) (when (not (z_19_0)) (not (v3_0))) (when (z_19_1) (v3_1)) (when (not (z_19_1)) (not (v3_1))) (when (z_19_2) (v3_2)) (when (not (z_19_2)) (not (v3_2))) (when (z_19_3) (v3_3)) (when (not (z_19_3)) (not (v3_3))) (when (z_19_4) (v3_4)) (when (not (z_19_4)) (not (v3_4))) (when (and (not (v3_4)) (z_19_4)) (of)) (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of))))
 (:action drink__guava__apple
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (when (z_21_0) (v5_0)) (when (not (z_21_0)) (not (v5_0))) (when (z_21_1) (v5_1)) (when (not (z_21_1)) (not (v5_1))) (when (z_21_2) (v5_2)) (when (not (z_21_2)) (not (v5_2))) (when (z_21_3) (v5_3)) (when (not (z_21_3)) (not (v5_3))) (when (z_21_4) (v5_4)) (when (not (z_21_4)) (not (v5_4))) (when (and (not (v5_4)) (z_21_4)) (of))))
 (:action drink__guava__hamburger
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_6_0) (v9_0)) (when (not (z_6_0)) (not (v9_0))) (when (z_6_1) (v9_1)) (when (not (z_6_1)) (not (v9_1))) (when (z_6_2) (v9_2)) (when (not (z_6_2)) (not (v9_2))) (when (z_6_3) (v9_3)) (when (not (z_6_3)) (not (v9_3))) (when (z_6_4) (v9_4)) (when (not (z_6_4)) (not (v9_4))) (when (and (not (v9_4)) (z_6_4)) (of)) (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of))))
 (:action drink__guava__haroset
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_0_0) (v6_0)) (when (not (z_0_0)) (not (v6_0))) (when (z_0_1) (v6_1)) (when (not (z_0_1)) (not (v6_1))) (when (z_0_2) (v6_2)) (when (not (z_0_2)) (not (v6_2))) (when (z_0_3) (v6_3)) (when (not (z_0_3)) (not (v6_3))) (when (z_0_4) (v6_4)) (when (not (z_0_4)) (not (v6_4))) (when (and (not (v6_4)) (z_0_4)) (of)) (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of))))
 (:action drink__guava__hotdog
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_18_0) (v2_0)) (when (not (z_18_0)) (not (v2_0))) (when (z_18_1) (v2_1)) (when (not (z_18_1)) (not (v2_1))) (when (z_18_2) (v2_2)) (when (not (z_18_2)) (not (v2_2))) (when (z_18_3) (v2_3)) (when (not (z_18_3)) (not (v2_3))) (when (z_18_4) (v2_4)) (when (not (z_18_4)) (not (v2_4))) (when (and (not (v2_4)) (z_18_4)) (of)) (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of))))
 (:action drink__guava__wurst
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (when (z_3_0) (v8_0)) (when (not (z_3_0)) (not (v8_0))) (when (z_3_1) (v8_1)) (when (not (z_3_1)) (not (v8_1))) (when (z_3_2) (v8_2)) (when (not (z_3_2)) (not (v8_2))) (when (z_3_3) (v8_3)) (when (not (z_3_3)) (not (v8_3))) (when (z_3_4) (v8_4)) (when (not (z_3_4)) (not (v8_4))) (when (and (not (v8_4)) (z_3_4)) (of))))
 (:action overcome__boils__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_guava) (craves_boils_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_boils_excitement) (not (craves_boils_guava))))
 (:action overcome__angina__excitement__guava
  :parameters ()
  :precondition (and (craves_angina_guava) (not (v4_4)) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_angina_guava)) (fears_angina_excitement)))
 (:action overcome__abrasion__excitement__haroset
  :parameters ()
  :precondition (and (craves_abrasion_haroset) (not (v4_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_abrasion_haroset)) (fears_abrasion_excitement)))
 (:action succumb__boils__excitement__guava
  :parameters ()
  :precondition (and (craves_excitement_guava) (fears_boils_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_boils_excitement)) (craves_boils_guava)))
 (:action succumb__boils__excitement__apple
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_boils_apple) (not (fears_boils_excitement))))
 (:action succumb__boils__excitement__haroset
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_boils_excitement)) (craves_boils_haroset)))
 (:action feast__curiosity__apple__guava
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (craves_curiosity_guava) (not (craves_curiosity_apple))))
 (:action succumb__anger__intoxication__hamburger
  :parameters ()
  :precondition (and (fears_anger_intoxication) (craves_intoxication_hamburger) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_anger_intoxication)) (craves_anger_hamburger)))
 (:action succumb__anger__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_anger_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_anger_haroset) (not (fears_anger_intoxication))))
 (:action succumb__anger__intoxication__guava
  :parameters ()
  :precondition (and (fears_anger_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_anger_guava) (not (fears_anger_intoxication))))
 (:action feast__curiosity__apple__hotdog
  :parameters ()
  :precondition (and (not (v5_4)) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (not (craves_curiosity_apple)) (craves_curiosity_hotdog)))
 (:action overcome__loneliness__excitement__haroset
  :parameters ()
  :precondition (and (not (v4_4)) (craves_loneliness_haroset) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_loneliness_excitement) (not (craves_loneliness_haroset))))
 (:action feast__curiosity__wurst__flounder
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (craves_curiosity_flounder) (not (craves_curiosity_wurst))))
 (:action feast__curiosity__wurst__hotdog
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (not (craves_curiosity_wurst)) (craves_curiosity_hotdog)))
 (:action overcome__hangover__excitement__haroset
  :parameters ()
  :precondition (and (not (v4_4)) (craves_hangover_haroset) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_hangover_haroset)) (fears_hangover_excitement)))
 (:action feast__entertainment__hamburger__haroset
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (not (v9_4)) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (not (craves_entertainment_hamburger)) (craves_entertainment_haroset)))
 (:action feast__entertainment__hamburger__flounder
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (not (v9_4)) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (craves_entertainment_flounder) (not (craves_entertainment_hamburger))))
 (:action feast__entertainment__wurst__flounder
  :parameters ()
  :precondition (and (craves_entertainment_wurst) (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (craves_entertainment_flounder) (not (craves_entertainment_wurst))))
 (:action feast__entertainment__wurst__hotdog
  :parameters ()
  :precondition (and (craves_entertainment_wurst) (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (not (craves_entertainment_wurst)) (craves_entertainment_hotdog)))
 (:action succumb__angina__excitement__guava
  :parameters ()
  :precondition (and (craves_excitement_guava) (fears_angina_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_angina_guava) (not (fears_angina_excitement))))
 (:action succumb__angina__excitement__apple
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_angina_apple) (not (fears_angina_excitement))))
 (:action succumb__angina__excitement__haroset
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_angina_haroset) (not (fears_angina_excitement))))
 (:action overcome__laceration__entertainment__wurst
  :parameters ()
  :precondition (and (not (v0_4)) (craves_laceration_wurst) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_laceration_wurst)) (fears_laceration_entertainment)))
 (:action overcome__laceration__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v10_4)) (craves_laceration_wurst) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_laceration_curiosity) (not (craves_laceration_wurst))))
 (:action overcome__jealousy__intoxication__guava
  :parameters ()
  :precondition (and (craves_jealousy_guava) (craves_intoxication_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_jealousy_guava)) (fears_jealousy_intoxication)))
 (:action succumb__prostatitis__intoxication__hamburger
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_hamburger) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_prostatitis_hamburger) (not (fears_prostatitis_intoxication))))
 (:action succumb__prostatitis__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_prostatitis_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_prostatitis_haroset) (not (fears_prostatitis_intoxication))))
 (:action succumb__prostatitis__intoxication__guava
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_prostatitis_guava) (not (fears_prostatitis_intoxication))))
 (:action overcome__grief__intoxication__guava
  :parameters ()
  :precondition (and (craves_grief_guava) (craves_intoxication_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_grief_guava)) (fears_grief_intoxication)))
 (:action succumb__sciatica__entertainment__wurst
  :parameters ()
  :precondition (and (fears_sciatica_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_sciatica_wurst) (not (fears_sciatica_entertainment))))
 (:action succumb__sciatica__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (fears_sciatica_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_sciatica_hamburger) (not (fears_sciatica_entertainment))))
 (:action succumb__sciatica__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_sciatica_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_sciatica_flounder) (not (fears_sciatica_entertainment))))
 (:action feast__excitement__apple__guava
  :parameters ()
  :precondition (and (not (v5_4)) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (craves_excitement_guava) (not (craves_excitement_apple))))
 (:action feast__excitement__apple__hotdog
  :parameters ()
  :precondition (and (not (v5_4)) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (craves_excitement_hotdog) (not (craves_excitement_apple))))
 (:action drink__hamburger__wurst
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (when (z_3_0) (v8_0)) (when (not (z_3_0)) (not (v8_0))) (when (z_3_1) (v8_1)) (when (not (z_3_1)) (not (v8_1))) (when (z_3_2) (v8_2)) (when (not (z_3_2)) (not (v8_2))) (when (z_3_3) (v8_3)) (when (not (z_3_3)) (not (v8_3))) (when (z_3_4) (v8_4)) (when (not (z_3_4)) (not (v8_4))) (when (and (not (v8_4)) (z_3_4)) (of))))
 (:action drink__hamburger__guava
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (when (z_1_0) (v7_0)) (when (not (z_1_0)) (not (v7_0))) (when (z_1_1) (v7_1)) (when (not (z_1_1)) (not (v7_1))) (when (z_1_2) (v7_2)) (when (not (z_1_2)) (not (v7_2))) (when (z_1_3) (v7_3)) (when (not (z_1_3)) (not (v7_3))) (when (z_1_4) (v7_4)) (when (not (z_1_4)) (not (v7_4))) (when (and (not (v7_4)) (z_1_4)) (of))))
 (:action drink__hamburger__hotdog
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_18_0) (v2_0)) (when (not (z_18_0)) (not (v2_0))) (when (z_18_1) (v2_1)) (when (not (z_18_1)) (not (v2_1))) (when (z_18_2) (v2_2)) (when (not (z_18_2)) (not (v2_2))) (when (z_18_3) (v2_3)) (when (not (z_18_3)) (not (v2_3))) (when (z_18_4) (v2_4)) (when (not (z_18_4)) (not (v2_4))) (when (and (not (v2_4)) (z_18_4)) (of)) (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of))))
 (:action drink__hamburger__apple
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_21_0) (v5_0)) (when (not (z_21_0)) (not (v5_0))) (when (z_21_1) (v5_1)) (when (not (z_21_1)) (not (v5_1))) (when (z_21_2) (v5_2)) (when (not (z_21_2)) (not (v5_2))) (when (z_21_3) (v5_3)) (when (not (z_21_3)) (not (v5_3))) (when (z_21_4) (v5_4)) (when (not (z_21_4)) (not (v5_4))) (when (and (not (v5_4)) (z_21_4)) (of)) (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of))))
 (:action drink__hamburger__haroset
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_0_0) (v6_0)) (when (not (z_0_0)) (not (v6_0))) (when (z_0_1) (v6_1)) (when (not (z_0_1)) (not (v6_1))) (when (z_0_2) (v6_2)) (when (not (z_0_2)) (not (v6_2))) (when (z_0_3) (v6_3)) (when (not (z_0_3)) (not (v6_3))) (when (z_0_4) (v6_4)) (when (not (z_0_4)) (not (v6_4))) (when (and (not (v6_4)) (z_0_4)) (of)) (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of))))
 (:action drink__hamburger__flounder
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_19_0) (v3_0)) (when (not (z_19_0)) (not (v3_0))) (when (z_19_1) (v3_1)) (when (not (z_19_1)) (not (v3_1))) (when (z_19_2) (v3_2)) (when (not (z_19_2)) (not (v3_2))) (when (z_19_3) (v3_3)) (when (not (z_19_3)) (not (v3_3))) (when (z_19_4) (v3_4)) (when (not (z_19_4)) (not (v3_4))) (when (and (not (v3_4)) (z_19_4)) (of)) (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of))))
 (:action feast__excitement__haroset__guava
  :parameters ()
  :precondition (and (not (v6_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (craves_excitement_guava) (not (craves_excitement_haroset))))
 (:action feast__excitement__haroset__hamburger
  :parameters ()
  :precondition (and (not (v6_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (not (craves_excitement_haroset)) (craves_excitement_hamburger)))
 (:action succumb__hangover__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_hangover_haroset) (not (fears_hangover_intoxication))))
 (:action succumb__hangover__intoxication__hamburger
  :parameters ()
  :precondition (and (fears_hangover_intoxication) (craves_intoxication_hamburger) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_hangover_hamburger) (not (fears_hangover_intoxication))))
 (:action succumb__hangover__intoxication__guava
  :parameters ()
  :precondition (and (fears_hangover_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_hangover_intoxication)) (craves_hangover_guava)))
 (:action overcome__prostatitis__excitement__haroset
  :parameters ()
  :precondition (and (craves_prostatitis_haroset) (not (v4_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_prostatitis_haroset)) (fears_prostatitis_excitement)))
 (:action feast__intoxication__hamburger__flounder
  :parameters ()
  :precondition (and (not (v9_4)) (craves_intoxication_hamburger) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (craves_intoxication_flounder) (not (craves_intoxication_hamburger))))
 (:action feast__intoxication__hamburger__haroset
  :parameters ()
  :precondition (and (not (v9_4)) (craves_intoxication_hamburger) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (craves_intoxication_haroset) (not (craves_intoxication_hamburger))))
 (:action succumb__grief__excitement__guava
  :parameters ()
  :precondition (and (craves_excitement_guava) (fears_grief_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_grief_guava) (not (fears_grief_excitement))))
 (:action succumb__grief__excitement__apple
  :parameters ()
  :precondition (and (fears_grief_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_grief_apple) (not (fears_grief_excitement))))
 (:action succumb__grief__excitement__haroset
  :parameters ()
  :precondition (and (fears_grief_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_grief_haroset) (not (fears_grief_excitement))))
 (:action overcome__anger__excitement__haroset
  :parameters ()
  :precondition (and (not (v4_4)) (craves_anger_haroset) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anger_haroset)) (fears_anger_excitement)))
 (:action overcome__anxiety__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v10_4)) (craves_anxiety_wurst) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_anxiety_wurst)) (fears_anxiety_curiosity)))
 (:action succumb__jealousy__excitement__guava
  :parameters ()
  :precondition (and (craves_excitement_guava) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_guava)))
 (:action feast__intoxication__guava__apple
  :parameters ()
  :precondition (and (not (v7_4)) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (not (craves_intoxication_guava)) (craves_intoxication_apple)))
 (:action succumb__jealousy__excitement__apple
  :parameters ()
  :precondition (and (fears_jealousy_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_apple)))
 (:action succumb__jealousy__excitement__haroset
  :parameters ()
  :precondition (and (fears_jealousy_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_jealousy_haroset) (not (fears_jealousy_excitement))))
 (:action feast__intoxication__guava__haroset
  :parameters ()
  :precondition (and (not (v7_4)) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (not (craves_intoxication_guava)) (craves_intoxication_haroset)))
 (:action overcome__boils__intoxication__guava
  :parameters ()
  :precondition (and (craves_intoxication_guava) (craves_boils_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_boils_intoxication) (not (craves_boils_guava))))
 (:action overcome__anxiety__entertainment__wurst
  :parameters ()
  :precondition (and (not (v0_4)) (craves_anxiety_wurst) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_anxiety_wurst)) (fears_anxiety_entertainment)))
 (:action overcome__angina__intoxication__guava
  :parameters ()
  :precondition (and (craves_angina_guava) (craves_intoxication_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_angina_guava)) (fears_angina_intoxication)))
 (:action succumb__loneliness__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_loneliness_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_haroset)))
 (:action succumb__loneliness__intoxication__hamburger
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_hamburger) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_hamburger)))
 (:action succumb__loneliness__intoxication__guava
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_guava)))
 (:action succumb__dread__entertainment__wurst
  :parameters ()
  :precondition (and (craves_entertainment_wurst) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_dread_wurst) (not (fears_dread_entertainment))))
 (:action succumb__dread__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_dread_flounder) (not (fears_dread_entertainment))))
 (:action succumb__dread__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_dread_hamburger) (not (fears_dread_entertainment))))
 (:action succumb__abrasion__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (fears_abrasion_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_abrasion_hamburger) (not (fears_abrasion_intoxication))))
 (:action succumb__abrasion__intoxication__guava
  :parameters ()
  :precondition (and (fears_abrasion_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_abrasion_guava) (not (fears_abrasion_intoxication))))
 (:action succumb__abrasion__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_abrasion_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_abrasion_haroset) (not (fears_abrasion_intoxication))))
 (:action succumb__anxiety__curiosity__apple
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_anxiety_curiosity)) (craves_anxiety_apple)))
 (:action succumb__anxiety__curiosity__flounder
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_anxiety_flounder) (not (fears_anxiety_curiosity))))
 (:action succumb__anxiety__curiosity__wurst
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_wurst) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_anxiety_curiosity)) (craves_anxiety_wurst)))
 (:action succumb__anxiety__curiosity__guava
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_anxiety_curiosity)) (craves_anxiety_guava)))
 (:action succumb__anxiety__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_anxiety_curiosity)) (craves_anxiety_hotdog)))
 (:action overcome__dread__entertainment__wurst
  :parameters ()
  :precondition (and (craves_dread_wurst) (not (v0_4)) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_dread_entertainment) (not (craves_dread_wurst))))
 (:action overcome__dread__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_dread_hamburger) (craves_entertainment_hamburger) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_dread_hamburger)) (fears_dread_entertainment)))
 (:action overcome__abrasion__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_abrasion_guava) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_abrasion_excitement) (not (craves_abrasion_guava))))
 (:action succumb__anxiety__entertainment__hotdog
  :parameters ()
  :precondition (and (fears_anxiety_entertainment) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_anxiety_entertainment)) (craves_anxiety_hotdog)))
 (:action overcome__abrasion__excitement__hamburger
  :parameters ()
  :precondition (and (craves_abrasion_hamburger) (not (v4_4)) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_abrasion_hamburger)) (fears_abrasion_excitement)))
 (:action succumb__anxiety__entertainment__haroset
  :parameters ()
  :precondition (and (fears_anxiety_entertainment) (craves_entertainment_haroset) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_anxiety_haroset) (not (fears_anxiety_entertainment))))
 (:action succumb__anxiety__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_anxiety_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_anxiety_flounder) (not (fears_anxiety_entertainment))))
 (:action succumb__anxiety__entertainment__wurst
  :parameters ()
  :precondition (and (fears_anxiety_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_anxiety_entertainment)) (craves_anxiety_wurst)))
 (:action succumb__anxiety__entertainment__hamburger
  :parameters ()
  :precondition (and (fears_anxiety_entertainment) (craves_entertainment_hamburger) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_anxiety_hamburger) (not (fears_anxiety_entertainment))))
 (:action overcome__dread__curiosity__flounder
  :parameters ()
  :precondition (and (craves_dread_flounder) (craves_curiosity_flounder) (not (v10_4)) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_dread_curiosity) (not (craves_dread_flounder))))
 (:action overcome__dread__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v10_4)) (craves_dread_wurst) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_dread_curiosity) (not (craves_dread_wurst))))
 (:action succumb__boils__excitement__hamburger
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_boils_excitement)) (craves_boils_hamburger)))
 (:action succumb__boils__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_boils_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_boils_hotdog) (not (fears_boils_excitement))))
 (:action overcome__dread__intoxication__flounder
  :parameters ()
  :precondition (and (craves_dread_flounder) (craves_intoxication_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_dread_intoxication) (not (craves_dread_flounder))))
 (:action overcome__dread__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_dread_hamburger) (craves_intoxication_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_dread_hamburger)) (fears_dread_intoxication)))
 (:action feast__curiosity__flounder__wurst
  :parameters ()
  :precondition (and (not (v3_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (craves_curiosity_wurst) (not (craves_curiosity_flounder))))
 (:action feast__curiosity__flounder__hamburger
  :parameters ()
  :precondition (and (not (v3_4)) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (craves_curiosity_hamburger) (not (craves_curiosity_flounder))))
 (:action overcome__abrasion__entertainment__haroset
  :parameters ()
  :precondition (and (craves_abrasion_haroset) (craves_entertainment_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_abrasion_haroset)) (fears_abrasion_entertainment)))
 (:action overcome__abrasion__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_abrasion_hamburger) (craves_entertainment_hamburger) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_abrasion_hamburger)) (fears_abrasion_entertainment)))
 (:action succumb__anger__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_anger_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_anger_flounder) (not (fears_anger_intoxication))))
 (:action succumb__anger__intoxication__apple
  :parameters ()
  :precondition (and (fears_anger_intoxication) (craves_intoxication_apple) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_anger_intoxication)) (craves_anger_apple)))
 (:action overcome__loneliness__excitement__hamburger
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_hamburger) (craves_loneliness_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_loneliness_hamburger)) (fears_loneliness_excitement)))
 (:action overcome__loneliness__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_guava) (craves_loneliness_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_loneliness_excitement) (not (craves_loneliness_guava))))
 (:action overcome__abrasion__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_abrasion_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_guava))))
 (:action overcome__abrasion__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_abrasion_hamburger) (craves_intoxication_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_abrasion_hamburger)) (fears_abrasion_intoxication)))
 (:action overcome__abrasion__intoxication__guava
  :parameters ()
  :precondition (and (craves_abrasion_guava) (craves_intoxication_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_abrasion_intoxication) (not (craves_abrasion_guava))))
 (:action overcome__loneliness__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (not (v0_4)) (craves_loneliness_haroset) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_loneliness_entertainment) (not (craves_loneliness_haroset))))
 (:action overcome__loneliness__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (not (v0_4)) (craves_loneliness_hamburger) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_loneliness_hamburger)) (fears_loneliness_entertainment)))
 (:action succumb__angina__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (fears_angina_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_angina_intoxication)) (craves_angina_hamburger)))
 (:action succumb__angina__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_angina_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_angina_haroset) (not (fears_angina_intoxication))))
 (:action succumb__angina__intoxication__guava
  :parameters ()
  :precondition (and (fears_angina_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_angina_guava) (not (fears_angina_intoxication))))
 (:action succumb__angina__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_angina_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_angina_flounder) (not (fears_angina_intoxication))))
 (:action succumb__angina__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (fears_angina_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_angina_apple) (not (fears_angina_intoxication))))
 (:action overcome__hangover__excitement__hamburger
  :parameters ()
  :precondition (and (not (v4_4)) (craves_hangover_hamburger) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_hangover_hamburger)) (fears_hangover_excitement)))
 (:action overcome__hangover__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_guava) (craves_hangover_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_hangover_guava)) (fears_hangover_excitement)))
 (:action overcome__loneliness__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_loneliness_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_guava))))
 (:action feast__curiosity__guava__haroset
  :parameters ()
  :precondition (and (not (v7_4)) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (craves_curiosity_haroset) (not (craves_curiosity_guava))))
 (:action feast__curiosity__guava__apple
  :parameters ()
  :precondition (and (not (v7_4)) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (craves_curiosity_apple) (not (craves_curiosity_guava))))
 (:action feast__entertainment__haroset__hamburger
  :parameters ()
  :precondition (and (not (v6_4)) (craves_entertainment_haroset) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (craves_entertainment_hamburger) (not (craves_entertainment_haroset))))
 (:action feast__entertainment__haroset__guava
  :parameters ()
  :precondition (and (not (v6_4)) (craves_entertainment_haroset) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (craves_entertainment_guava) (not (craves_entertainment_haroset))))
 (:action overcome__loneliness__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (not (v1_4)) (craves_loneliness_hamburger) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_loneliness_hamburger)) (fears_loneliness_intoxication)))
 (:action overcome__loneliness__intoxication__guava
  :parameters ()
  :precondition (and (craves_loneliness_guava) (craves_intoxication_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_loneliness_guava)) (fears_loneliness_intoxication)))
 (:action feast__entertainment__hotdog__apple
  :parameters ()
  :precondition (and (not (v2_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (craves_entertainment_apple) (not (craves_entertainment_hotdog))))
 (:action feast__entertainment__hotdog__wurst
  :parameters ()
  :precondition (and (not (v2_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (craves_entertainment_wurst) (not (craves_entertainment_hotdog))))
 (:action overcome__hangover__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (craves_hangover_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_hangover_haroset)) (fears_hangover_entertainment)))
 (:action overcome__hangover__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (not (v0_4)) (craves_hangover_hamburger) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_hangover_entertainment) (not (craves_hangover_hamburger))))
 (:action succumb__boils__intoxication__guava
  :parameters ()
  :precondition (and (craves_intoxication_guava) (fears_boils_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_guava)))
 (:action succumb__boils__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_boils_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_flounder)))
 (:action succumb__boils__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (fears_boils_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_boils_apple) (not (fears_boils_intoxication))))
 (:action succumb__boils__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (fears_boils_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_hamburger)))
 (:action succumb__boils__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_boils_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_boils_haroset) (not (fears_boils_intoxication))))
 (:action overcome__hangover__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_curiosity_guava) (craves_hangover_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_hangover_curiosity) (not (craves_hangover_guava))))
 (:action succumb__angina__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_angina_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_angina_excitement)) (craves_angina_hotdog)))
 (:action succumb__angina__excitement__hamburger
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_angina_excitement)) (craves_angina_hamburger)))
 (:action overcome__hangover__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (craves_hangover_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_hamburger))))
 (:action overcome__hangover__intoxication__guava
  :parameters ()
  :precondition (and (craves_intoxication_guava) (not (v1_4)) (craves_hangover_guava) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_guava))))
 (:action overcome__jealousy__excitement__apple
  :parameters ()
  :precondition (and (not (v4_4)) (craves_jealousy_apple) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_jealousy_excitement) (not (craves_jealousy_apple))))
 (:action overcome__jealousy__excitement__haroset
  :parameters ()
  :precondition (and (craves_jealousy_haroset) (not (v4_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_jealousy_haroset)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__entertainment__haroset
  :parameters ()
  :precondition (and (craves_jealousy_haroset) (craves_entertainment_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_jealousy_haroset)) (fears_jealousy_entertainment)))
 (:action overcome__grief__excitement__apple
  :parameters ()
  :precondition (and (craves_grief_apple) (not (v4_4)) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_grief_apple)) (fears_grief_excitement)))
 (:action overcome__grief__excitement__haroset
  :parameters ()
  :precondition (and (craves_grief_haroset) (not (v4_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_grief_excitement) (not (craves_grief_haroset))))
 (:action overcome__jealousy__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_jealousy_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_jealousy_curiosity) (not (craves_jealousy_guava))))
 (:action overcome__jealousy__curiosity__apple
  :parameters ()
  :precondition (and (not (v10_4)) (craves_jealousy_apple) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_jealousy_curiosity) (not (craves_jealousy_apple))))
 (:action overcome__jealousy__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (craves_jealousy_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_jealousy_intoxication) (not (craves_jealousy_apple))))
 (:action overcome__jealousy__intoxication__haroset
  :parameters ()
  :precondition (and (craves_jealousy_haroset) (craves_intoxication_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_jealousy_haroset)) (fears_jealousy_intoxication)))
 (:action overcome__grief__entertainment__haroset
  :parameters ()
  :precondition (and (craves_grief_haroset) (craves_entertainment_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_grief_entertainment) (not (craves_grief_haroset))))
 (:action succumb__prostatitis__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_prostatitis_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_prostatitis_intoxication)) (craves_prostatitis_flounder)))
 (:action succumb__prostatitis__intoxication__apple
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_apple) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_prostatitis_apple) (not (fears_prostatitis_intoxication))))
 (:action overcome__grief__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_grief_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_grief_curiosity) (not (craves_grief_guava))))
 (:action overcome__grief__curiosity__apple
  :parameters ()
  :precondition (and (craves_grief_apple) (not (v10_4)) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_grief_curiosity) (not (craves_grief_apple))))
 (:action succumb__anger__excitement__hamburger
  :parameters ()
  :precondition (and (fears_anger_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anger_hamburger) (not (fears_anger_excitement))))
 (:action succumb__anger__excitement__haroset
  :parameters ()
  :precondition (and (fears_anger_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anger_haroset) (not (fears_anger_excitement))))
 (:action succumb__anger__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_anger_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anger_hotdog) (not (fears_anger_excitement))))
 (:action succumb__anger__excitement__guava
  :parameters ()
  :precondition (and (fears_anger_excitement) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anger_guava) (not (fears_anger_excitement))))
 (:action overcome__grief__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (craves_grief_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_grief_intoxication) (not (craves_grief_haroset))))
 (:action succumb__anger__excitement__apple
  :parameters ()
  :precondition (and (fears_anger_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anger_apple) (not (fears_anger_excitement))))
 (:action overcome__grief__intoxication__apple
  :parameters ()
  :precondition (and (craves_grief_apple) (craves_intoxication_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_grief_apple)) (fears_grief_intoxication)))
 (:action succumb__sciatica__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_entertainment_hotdog) (fears_sciatica_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_sciatica_hotdog) (not (fears_sciatica_entertainment))))
 (:action succumb__sciatica__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (fears_sciatica_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_sciatica_haroset) (not (fears_sciatica_entertainment))))
 (:action succumb__prostatitis__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_prostatitis_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_prostatitis_hotdog) (not (fears_prostatitis_excitement))))
 (:action succumb__prostatitis__excitement__guava
  :parameters ()
  :precondition (and (fears_prostatitis_excitement) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_prostatitis_guava) (not (fears_prostatitis_excitement))))
 (:action succumb__prostatitis__excitement__apple
  :parameters ()
  :precondition (and (fears_prostatitis_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_prostatitis_apple) (not (fears_prostatitis_excitement))))
 (:action succumb__prostatitis__excitement__hamburger
  :parameters ()
  :precondition (and (fears_prostatitis_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_prostatitis_hamburger) (not (fears_prostatitis_excitement))))
 (:action succumb__prostatitis__excitement__haroset
  :parameters ()
  :precondition (and (fears_prostatitis_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_prostatitis_haroset) (not (fears_prostatitis_excitement))))
 (:action overcome__sciatica__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_sciatica_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_flounder))))
 (:action overcome__sciatica__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (craves_sciatica_wurst) (not (v10_4)) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_sciatica_wurst)) (fears_sciatica_curiosity)))
 (:action succumb__laceration__curiosity__guava
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_laceration_guava) (not (fears_laceration_curiosity))))
 (:action overcome__sciatica__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (craves_sciatica_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_sciatica_flounder)) (fears_sciatica_intoxication)))
 (:action overcome__sciatica__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_sciatica_hamburger) (craves_intoxication_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_sciatica_hamburger)) (fears_sciatica_intoxication)))
 (:action succumb__laceration__curiosity__flounder
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_laceration_flounder) (not (fears_laceration_curiosity))))
 (:action succumb__laceration__curiosity__apple
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_laceration_apple) (not (fears_laceration_curiosity))))
 (:action succumb__laceration__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_laceration_hotdog) (not (fears_laceration_curiosity))))
 (:action succumb__laceration__curiosity__wurst
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_wurst) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_laceration_wurst) (not (fears_laceration_curiosity))))
 (:action succumb__laceration__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_entertainment_hotdog) (fears_laceration_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_laceration_hotdog) (not (fears_laceration_entertainment))))
 (:action succumb__laceration__entertainment__wurst
  :parameters ()
  :precondition (and (fears_laceration_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_laceration_wurst) (not (fears_laceration_entertainment))))
 (:action overcome__sciatica__excitement__hamburger
  :parameters ()
  :precondition (and (craves_sciatica_hamburger) (not (v4_4)) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_hamburger))))
 (:action feast__excitement__hamburger__haroset
  :parameters ()
  :precondition (and (not (v9_4)) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (craves_excitement_haroset) (not (craves_excitement_hamburger))))
 (:action feast__excitement__hamburger__flounder
  :parameters ()
  :precondition (and (not (v9_4)) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (craves_excitement_flounder) (not (craves_excitement_hamburger))))
 (:action overcome__prostatitis__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_prostatitis_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_prostatitis_guava)) (fears_prostatitis_curiosity)))
 (:action succumb__hangover__excitement__haroset
  :parameters ()
  :precondition (and (fears_hangover_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_hangover_excitement)) (craves_hangover_haroset)))
 (:action succumb__hangover__excitement__hamburger
  :parameters ()
  :precondition (and (fears_hangover_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_hangover_excitement)) (craves_hangover_hamburger)))
 (:action succumb__hangover__excitement__guava
  :parameters ()
  :precondition (and (fears_hangover_excitement) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_hangover_excitement)) (craves_hangover_guava)))
 (:action succumb__hangover__excitement__hotdog
  :parameters ()
  :precondition (and (fears_hangover_excitement) (craves_excitement_hotdog) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_hangover_excitement)) (craves_hangover_hotdog)))
 (:action succumb__hangover__excitement__apple
  :parameters ()
  :precondition (and (fears_hangover_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_hangover_excitement)) (craves_hangover_apple)))
 (:action overcome__prostatitis__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_prostatitis_hamburger) (craves_intoxication_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_prostatitis_hamburger)) (fears_prostatitis_intoxication)))
 (:action overcome__prostatitis__intoxication__guava
  :parameters ()
  :precondition (and (craves_intoxication_guava) (craves_prostatitis_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_prostatitis_intoxication) (not (craves_prostatitis_guava))))
 (:action feast__excitement__hotdog__apple
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_hotdog) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (craves_excitement_apple) (not (craves_excitement_hotdog))))
 (:action feast__intoxication__flounder__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (not (v3_4)) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (not (craves_intoxication_flounder)) (craves_intoxication_hamburger)))
 (:action feast__excitement__hotdog__wurst
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_hotdog) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (craves_excitement_wurst) (not (craves_excitement_hotdog))))
 (:action feast__intoxication__flounder__wurst
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (not (v3_4)) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (not (craves_intoxication_flounder)) (craves_intoxication_wurst)))
 (:action overcome__sciatica__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_sciatica_hamburger) (craves_entertainment_hamburger) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_sciatica_hamburger)) (fears_sciatica_entertainment)))
 (:action overcome__sciatica__entertainment__wurst
  :parameters ()
  :precondition (and (craves_sciatica_wurst) (not (v0_4)) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_sciatica_wurst)) (fears_sciatica_entertainment)))
 (:action succumb__hangover__intoxication__apple
  :parameters ()
  :precondition (and (fears_hangover_intoxication) (craves_intoxication_apple) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_hangover_intoxication)) (craves_hangover_apple)))
 (:action succumb__hangover__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_hangover_intoxication)) (craves_hangover_flounder)))
 (:action feast__intoxication__apple__guava
  :parameters ()
  :precondition (and (not (v5_4)) (craves_intoxication_apple) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (not (craves_intoxication_apple)) (craves_intoxication_guava)))
 (:action feast__intoxication__apple__hotdog
  :parameters ()
  :precondition (and (not (v5_4)) (craves_intoxication_apple) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (craves_intoxication_hotdog) (not (craves_intoxication_apple))))
 (:action overcome__prostatitis__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_guava) (craves_prostatitis_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_prostatitis_guava)) (fears_prostatitis_excitement)))
 (:action overcome__prostatitis__excitement__hamburger
  :parameters ()
  :precondition (and (craves_prostatitis_hamburger) (not (v4_4)) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_prostatitis_hamburger)) (fears_prostatitis_excitement)))
 (:action overcome__anger__curiosity__guava
  :parameters ()
  :precondition (and (craves_anger_guava) (not (v10_4)) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_anger_guava)) (fears_anger_curiosity)))
 (:action succumb__grief__excitement__hamburger
  :parameters ()
  :precondition (and (fears_grief_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_grief_excitement)) (craves_grief_hamburger)))
 (:action succumb__grief__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_grief_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_grief_hotdog) (not (fears_grief_excitement))))
 (:action overcome__anger__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (craves_anger_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_anger_intoxication) (not (craves_anger_hamburger))))
 (:action overcome__anger__intoxication__guava
  :parameters ()
  :precondition (and (craves_anger_guava) (craves_intoxication_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_anger_guava)) (fears_anger_intoxication)))
 (:action overcome__prostatitis__entertainment__haroset
  :parameters ()
  :precondition (and (craves_prostatitis_haroset) (craves_entertainment_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_prostatitis_haroset)) (fears_prostatitis_entertainment)))
 (:action overcome__prostatitis__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_prostatitis_hamburger) (craves_entertainment_hamburger) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_prostatitis_hamburger)) (fears_prostatitis_entertainment)))
 (:action succumb__grief__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_grief_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_grief_haroset) (not (fears_grief_intoxication))))
 (:action succumb__grief__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_grief_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_grief_flounder) (not (fears_grief_intoxication))))
 (:action succumb__grief__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (fears_grief_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_grief_intoxication)) (craves_grief_hamburger)))
 (:action succumb__grief__intoxication__guava
  :parameters ()
  :precondition (and (fears_grief_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_grief_guava) (not (fears_grief_intoxication))))
 (:action succumb__grief__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (fears_grief_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_grief_apple) (not (fears_grief_intoxication))))
 (:action overcome__anger__excitement__hamburger
  :parameters ()
  :precondition (and (not (v4_4)) (craves_anger_hamburger) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_anger_excitement) (not (craves_anger_hamburger))))
 (:action overcome__anger__excitement__guava
  :parameters ()
  :precondition (and (craves_anger_guava) (not (v4_4)) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anger_guava)) (fears_anger_excitement)))
 (:action succumb__laceration__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_laceration_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_laceration_flounder) (not (fears_laceration_entertainment))))
 (:action succumb__laceration__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (fears_laceration_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_laceration_hamburger) (not (fears_laceration_entertainment))))
 (:action succumb__laceration__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (fears_laceration_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_laceration_haroset) (not (fears_laceration_entertainment))))
 (:action succumb__jealousy__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_hotdog)))
 (:action succumb__jealousy__excitement__hamburger
  :parameters ()
  :precondition (and (fears_jealousy_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_hamburger)))
 (:action overcome__anger__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (not (v0_4)) (craves_anger_haroset) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anger_entertainment) (not (craves_anger_haroset))))
 (:action overcome__anger__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (not (v0_4)) (craves_anger_hamburger) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anger_entertainment) (not (craves_anger_hamburger))))
 (:action succumb__jealousy__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_jealousy_hamburger) (not (fears_jealousy_intoxication))))
 (:action succumb__jealousy__intoxication__guava
  :parameters ()
  :precondition (and (fears_jealousy_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_jealousy_guava) (not (fears_jealousy_intoxication))))
 (:action succumb__jealousy__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_jealousy_apple) (not (fears_jealousy_intoxication))))
 (:action succumb__jealousy__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_jealousy_haroset) (not (fears_jealousy_intoxication))))
 (:action succumb__jealousy__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_jealousy_flounder) (not (fears_jealousy_intoxication))))
 (:action overcome__boils__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_boils_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_boils_curiosity) (not (craves_boils_guava))))
 (:action overcome__boils__curiosity__apple
  :parameters ()
  :precondition (and (not (v10_4)) (craves_boils_apple) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_boils_apple)) (fears_boils_curiosity)))
 (:action overcome__boils__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (craves_boils_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_boils_apple)) (fears_boils_intoxication)))
 (:action overcome__boils__intoxication__haroset
  :parameters ()
  :precondition (and (craves_boils_haroset) (craves_intoxication_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_boils_intoxication) (not (craves_boils_haroset))))
 (:action overcome__boils__excitement__apple
  :parameters ()
  :precondition (and (not (v4_4)) (craves_boils_apple) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_boils_excitement) (not (craves_boils_apple))))
 (:action overcome__boils__excitement__haroset
  :parameters ()
  :precondition (and (craves_boils_haroset) (not (v4_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_boils_excitement) (not (craves_boils_haroset))))
 (:action overcome__angina__curiosity__guava
  :parameters ()
  :precondition (and (craves_angina_guava) (not (v10_4)) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_angina_curiosity) (not (craves_angina_guava))))
 (:action overcome__angina__curiosity__apple
  :parameters ()
  :precondition (and (craves_angina_apple) (not (v10_4)) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_angina_curiosity) (not (craves_angina_apple))))
 (:action overcome__angina__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (craves_angina_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_angina_haroset)) (fears_angina_intoxication)))
 (:action overcome__angina__intoxication__apple
  :parameters ()
  :precondition (and (craves_angina_apple) (craves_intoxication_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_angina_apple)) (fears_angina_intoxication)))
 (:action overcome__boils__entertainment__haroset
  :parameters ()
  :precondition (and (craves_boils_haroset) (craves_entertainment_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_boils_haroset)) (fears_boils_entertainment)))
 (:action overcome__angina__excitement__apple
  :parameters ()
  :precondition (and (craves_angina_apple) (not (v4_4)) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_angina_apple)) (fears_angina_excitement)))
 (:action overcome__angina__excitement__haroset
  :parameters ()
  :precondition (and (not (v4_4)) (craves_angina_haroset) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_angina_excitement) (not (craves_angina_haroset))))
 (:action succumb__loneliness__excitement__hamburger
  :parameters ()
  :precondition (and (fears_loneliness_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_loneliness_hamburger) (not (fears_loneliness_excitement))))
 (:action succumb__loneliness__excitement__guava
  :parameters ()
  :precondition (and (fears_loneliness_excitement) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_loneliness_guava) (not (fears_loneliness_excitement))))
 (:action succumb__loneliness__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_loneliness_hotdog) (not (fears_loneliness_excitement))))
 (:action succumb__loneliness__excitement__apple
  :parameters ()
  :precondition (and (fears_loneliness_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_loneliness_excitement)) (craves_loneliness_apple)))
 (:action succumb__loneliness__excitement__haroset
  :parameters ()
  :precondition (and (fears_loneliness_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_loneliness_excitement)) (craves_loneliness_haroset)))
 (:action overcome__angina__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (not (v0_4)) (craves_angina_haroset) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_angina_entertainment) (not (craves_angina_haroset))))
 (:action succumb__loneliness__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_loneliness_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_flounder)))
 (:action succumb__loneliness__intoxication__apple
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_apple) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_apple)))
 (:action succumb__dread__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_entertainment_hotdog) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_dread_entertainment)) (craves_dread_hotdog)))
 (:action succumb__dread__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_dread_haroset) (not (fears_dread_entertainment))))
 (:action succumb__abrasion__excitement__guava
  :parameters ()
  :precondition (and (craves_excitement_guava) (fears_abrasion_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_abrasion_guava) (not (fears_abrasion_excitement))))
 (:action succumb__abrasion__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_abrasion_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_abrasion_hotdog) (not (fears_abrasion_excitement))))
 (:action succumb__abrasion__excitement__apple
  :parameters ()
  :precondition (and (fears_abrasion_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_abrasion_apple) (not (fears_abrasion_excitement))))
 (:action succumb__abrasion__excitement__haroset
  :parameters ()
  :precondition (and (fears_abrasion_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_abrasion_haroset) (not (fears_abrasion_excitement))))
 (:action succumb__abrasion__excitement__hamburger
  :parameters ()
  :precondition (and (fears_abrasion_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_abrasion_hamburger) (not (fears_abrasion_excitement))))
 (:action overcome__dread__excitement__hamburger
  :parameters ()
  :precondition (and (craves_dread_hamburger) (not (v4_4)) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_dread_hamburger)) (fears_dread_excitement)))
 (:action succumb__abrasion__intoxication__apple
  :parameters ()
  :precondition (and (fears_abrasion_intoxication) (craves_intoxication_apple) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_abrasion_apple) (not (fears_abrasion_intoxication))))
 (:action succumb__abrasion__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_abrasion_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_abrasion_flounder) (not (fears_abrasion_intoxication))))
 (:action succumb__anxiety__curiosity__haroset
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_anxiety_curiosity)) (craves_anxiety_haroset)))
 (:action succumb__anxiety__curiosity__hamburger
  :parameters ()
  :precondition (and (fears_anxiety_curiosity) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_anxiety_hamburger) (not (fears_anxiety_curiosity))))
 (:action overcome__dread__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_dread_hotdog) (not (v0_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_dread_entertainment) (not (craves_dread_hotdog))))
 (:action overcome__dread__entertainment__haroset
  :parameters ()
  :precondition (and (craves_dread_haroset) (craves_entertainment_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_dread_haroset)) (fears_dread_entertainment)))
 (:action overcome__abrasion__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (craves_abrasion_hotdog) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_abrasion_hotdog)) (fears_abrasion_excitement)))
 (:action succumb__anxiety__entertainment__guava
  :parameters ()
  :precondition (and (fears_anxiety_entertainment) (craves_entertainment_guava) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_anxiety_guava) (not (fears_anxiety_entertainment))))
 (:action overcome__abrasion__excitement__apple
  :parameters ()
  :precondition (and (not (v4_4)) (craves_abrasion_apple) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_abrasion_excitement) (not (craves_abrasion_apple))))
 (:action overcome__abrasion__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (not (v4_4)) (craves_abrasion_flounder) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_abrasion_excitement) (not (craves_abrasion_flounder))))
 (:action succumb__anxiety__entertainment__apple
  :parameters ()
  :precondition (and (fears_anxiety_entertainment) (craves_entertainment_apple) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_anxiety_entertainment)) (craves_anxiety_apple)))
 (:action overcome__dread__curiosity__haroset
  :parameters ()
  :precondition (and (craves_dread_haroset) (not (v10_4)) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_dread_haroset)) (fears_dread_curiosity)))
 (:action overcome__dread__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_dread_hamburger) (not (v10_4)) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_dread_hamburger)) (fears_dread_curiosity)))
 (:action overcome__dread__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_dread_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_dread_curiosity) (not (craves_dread_hotdog))))
 (:action succumb__boils__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_boils_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_boils_excitement)) (craves_boils_flounder)))
 (:action succumb__boils__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_boils_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_boils_excitement)) (craves_boils_wurst)))
 (:action overcome__dread__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_dread_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_dread_intoxication) (not (craves_dread_hotdog))))
 (:action succumb__boils__curiosity__guava
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_boils_curiosity)) (craves_boils_guava)))
 (:action succumb__boils__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_boils_hotdog) (not (fears_boils_curiosity))))
 (:action overcome__dread__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (craves_dread_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_dread_haroset)) (fears_dread_intoxication)))
 (:action overcome__dread__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_dread_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_dread_intoxication) (not (craves_dread_wurst))))
 (:action succumb__anger__curiosity__haroset
  :parameters ()
  :precondition (and (fears_anger_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_anger_haroset) (not (fears_anger_curiosity))))
 (:action succumb__anger__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (fears_anger_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_anger_flounder) (not (fears_anger_curiosity))))
 (:action succumb__anger__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_anger_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_anger_wurst) (not (fears_anger_curiosity))))
 (:action succumb__anger__curiosity__hamburger
  :parameters ()
  :precondition (and (fears_anger_curiosity) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_anger_curiosity)) (craves_anger_hamburger)))
 (:action succumb__anger__curiosity__guava
  :parameters ()
  :precondition (and (fears_anger_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_anger_guava) (not (fears_anger_curiosity))))
 (:action succumb__anger__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_anger_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_anger_hotdog) (not (fears_anger_curiosity))))
 (:action succumb__anger__curiosity__apple
  :parameters ()
  :precondition (and (fears_anger_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_anger_apple) (not (fears_anger_curiosity))))
 (:action overcome__abrasion__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_abrasion_hotdog) (not (v0_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_abrasion_hotdog)) (fears_abrasion_entertainment)))
 (:action overcome__abrasion__entertainment__guava
  :parameters ()
  :precondition (and (craves_abrasion_guava) (craves_entertainment_guava) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_abrasion_guava)) (fears_abrasion_entertainment)))
 (:action overcome__abrasion__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (not (v0_4)) (craves_abrasion_apple) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_abrasion_apple)) (fears_abrasion_entertainment)))
 (:action overcome__abrasion__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_abrasion_flounder) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_abrasion_flounder)) (fears_abrasion_entertainment)))
 (:action succumb__anger__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_anger_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_anger_hotdog) (not (fears_anger_intoxication))))
 (:action succumb__anger__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_anger_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_anger_intoxication)) (craves_anger_wurst)))
 (:action overcome__loneliness__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (craves_loneliness_hotdog) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_loneliness_excitement) (not (craves_loneliness_hotdog))))
 (:action overcome__loneliness__excitement__apple
  :parameters ()
  :precondition (and (craves_loneliness_apple) (not (v4_4)) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_loneliness_excitement) (not (craves_loneliness_apple))))
 (:action overcome__loneliness__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (not (v4_4)) (craves_loneliness_flounder) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_loneliness_excitement) (not (craves_loneliness_flounder))))
 (:action succumb__anger__entertainment__apple
  :parameters ()
  :precondition (and (fears_anger_entertainment) (craves_entertainment_apple) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_apple)))
 (:action succumb__anger__entertainment__haroset
  :parameters ()
  :precondition (and (fears_anger_entertainment) (craves_entertainment_haroset) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_anger_haroset) (not (fears_anger_entertainment))))
 (:action succumb__anger__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_anger_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_anger_flounder) (not (fears_anger_entertainment))))
 (:action succumb__anger__entertainment__wurst
  :parameters ()
  :precondition (and (fears_anger_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_wurst)))
 (:action succumb__anger__entertainment__hamburger
  :parameters ()
  :precondition (and (fears_anger_entertainment) (craves_entertainment_hamburger) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_anger_entertainment)) (craves_anger_hamburger)))
 (:action succumb__anger__entertainment__guava
  :parameters ()
  :precondition (and (fears_anger_entertainment) (craves_entertainment_guava) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_anger_guava) (not (fears_anger_entertainment))))
 (:action succumb__anger__entertainment__hotdog
  :parameters ()
  :precondition (and (fears_anger_entertainment) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_anger_hotdog) (not (fears_anger_entertainment))))
 (:action feast__curiosity__hamburger__haroset
  :parameters ()
  :precondition (and (not (v9_4)) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (craves_curiosity_haroset) (not (craves_curiosity_hamburger))))
 (:action feast__curiosity__hamburger__flounder
  :parameters ()
  :precondition (and (not (v9_4)) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_14_0) (v9_0)) (when (not (z_14_0)) (not (v9_0))) (when (z_14_1) (v9_1)) (when (not (z_14_1)) (not (v9_1))) (when (z_14_2) (v9_2)) (when (not (z_14_2)) (not (v9_2))) (when (z_14_3) (v9_3)) (when (not (z_14_3)) (not (v9_3))) (when (z_14_4) (v9_4)) (when (not (z_14_4)) (not (v9_4))) (when (and (v9_4) (not (z_14_4))) (of)) (craves_curiosity_flounder) (not (craves_curiosity_hamburger))))
 (:action overcome__abrasion__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_abrasion_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_hotdog))))
 (:action overcome__abrasion__curiosity__apple
  :parameters ()
  :precondition (and (not (v10_4)) (craves_curiosity_apple) (craves_abrasion_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_apple))))
 (:action overcome__abrasion__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_abrasion_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_flounder))))
 (:action overcome__abrasion__curiosity__haroset
  :parameters ()
  :precondition (and (craves_abrasion_haroset) (not (v10_4)) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_haroset))))
 (:action overcome__abrasion__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_abrasion_hamburger) (not (v10_4)) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_hamburger))))
 (:action feast__curiosity__haroset__hamburger
  :parameters ()
  :precondition (and (not (v6_4)) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (not (craves_curiosity_haroset)) (craves_curiosity_hamburger)))
 (:action feast__curiosity__haroset__guava
  :parameters ()
  :precondition (and (not (v6_4)) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_11_0) (v6_0)) (when (not (z_11_0)) (not (v6_0))) (when (z_11_1) (v6_1)) (when (not (z_11_1)) (not (v6_1))) (when (z_11_2) (v6_2)) (when (not (z_11_2)) (not (v6_2))) (when (z_11_3) (v6_3)) (when (not (z_11_3)) (not (v6_3))) (when (z_11_4) (v6_4)) (when (not (z_11_4)) (not (v6_4))) (when (and (v6_4) (not (z_11_4))) (of)) (not (craves_curiosity_haroset)) (craves_curiosity_guava)))
 (:action overcome__abrasion__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_abrasion_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_abrasion_hotdog)) (fears_abrasion_intoxication)))
 (:action overcome__abrasion__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (craves_abrasion_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_abrasion_intoxication) (not (craves_abrasion_apple))))
 (:action overcome__abrasion__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (not (v1_4)) (craves_abrasion_flounder) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_abrasion_intoxication) (not (craves_abrasion_flounder))))
 (:action succumb__angina__curiosity__apple
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_angina_apple) (not (fears_angina_curiosity))))
 (:action succumb__angina__curiosity__haroset
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_haroset)))
 (:action succumb__angina__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (fears_angina_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_flounder)))
 (:action overcome__loneliness__entertainment__apple
  :parameters ()
  :precondition (and (craves_loneliness_apple) (craves_entertainment_apple) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_loneliness_entertainment) (not (craves_loneliness_apple))))
 (:action overcome__loneliness__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_loneliness_flounder) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_loneliness_entertainment) (not (craves_loneliness_flounder))))
 (:action overcome__loneliness__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_loneliness_hotdog) (not (v0_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_loneliness_entertainment) (not (craves_loneliness_hotdog))))
 (:action overcome__loneliness__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (not (v0_4)) (craves_loneliness_guava) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_loneliness_entertainment) (not (craves_loneliness_guava))))
 (:action succumb__angina__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_angina_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_angina_hotdog) (not (fears_angina_intoxication))))
 (:action succumb__angina__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_angina_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_angina_wurst) (not (fears_angina_intoxication))))
 (:action feast__entertainment__apple__guava
  :parameters ()
  :precondition (and (not (v5_4)) (craves_entertainment_apple) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (not (craves_entertainment_apple)) (craves_entertainment_guava)))
 (:action feast__entertainment__apple__hotdog
  :parameters ()
  :precondition (and (not (v5_4)) (craves_entertainment_apple) (not (of)))
  :effect (and (when (z_10_0) (v5_0)) (when (not (z_10_0)) (not (v5_0))) (when (z_10_1) (v5_1)) (when (not (z_10_1)) (not (v5_1))) (when (z_10_2) (v5_2)) (when (not (z_10_2)) (not (v5_2))) (when (z_10_3) (v5_3)) (when (not (z_10_3)) (not (v5_3))) (when (z_10_4) (v5_4)) (when (not (z_10_4)) (not (v5_4))) (when (and (v5_4) (not (z_10_4))) (of)) (not (craves_entertainment_apple)) (craves_entertainment_hotdog)))
 (:action overcome__hangover__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (craves_hangover_flounder) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_hangover_flounder)) (fears_hangover_excitement)))
 (:action overcome__hangover__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (not (v4_4)) (craves_hangover_hotdog) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_hangover_hotdog)) (fears_hangover_excitement)))
 (:action succumb__angina__entertainment__haroset
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_haroset) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_haroset)))
 (:action succumb__angina__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_angina_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_flounder)))
 (:action succumb__angina__entertainment__wurst
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_wurst)))
 (:action succumb__angina__entertainment__hamburger
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_hamburger) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_hamburger)))
 (:action succumb__angina__entertainment__guava
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_guava) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_angina_guava) (not (fears_angina_entertainment))))
 (:action succumb__angina__entertainment__hotdog
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_angina_entertainment)) (craves_angina_hotdog)))
 (:action overcome__hangover__excitement__apple
  :parameters ()
  :precondition (and (craves_hangover_apple) (not (v4_4)) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_hangover_apple)) (fears_hangover_excitement)))
 (:action succumb__angina__entertainment__apple
  :parameters ()
  :precondition (and (fears_angina_entertainment) (craves_entertainment_apple) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_angina_apple) (not (fears_angina_entertainment))))
 (:action overcome__loneliness__curiosity__haroset
  :parameters ()
  :precondition (and (not (v10_4)) (craves_loneliness_haroset) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_haroset))))
 (:action overcome__loneliness__curiosity__hamburger
  :parameters ()
  :precondition (and (not (v10_4)) (craves_curiosity_hamburger) (craves_loneliness_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_hamburger))))
 (:action overcome__loneliness__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_loneliness_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_hotdog))))
 (:action overcome__loneliness__curiosity__apple
  :parameters ()
  :precondition (and (craves_loneliness_apple) (not (v10_4)) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_apple))))
 (:action overcome__loneliness__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_loneliness_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_flounder))))
 (:action overcome__loneliness__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (craves_loneliness_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_loneliness_flounder)) (fears_loneliness_intoxication)))
 (:action overcome__loneliness__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_loneliness_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_loneliness_hotdog)) (fears_loneliness_intoxication)))
 (:action overcome__loneliness__intoxication__apple
  :parameters ()
  :precondition (and (craves_loneliness_apple) (craves_intoxication_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_loneliness_apple)) (fears_loneliness_intoxication)))
 (:action succumb__boils__curiosity__apple
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_boils_apple) (not (fears_boils_curiosity))))
 (:action succumb__boils__curiosity__haroset
  :parameters ()
  :precondition (and (fears_boils_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_boils_haroset) (not (fears_boils_curiosity))))
 (:action succumb__boils__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (fears_boils_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_boils_curiosity)) (craves_boils_flounder)))
 (:action succumb__boils__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_boils_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_boils_wurst) (not (fears_boils_curiosity))))
 (:action succumb__boils__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_curiosity_hamburger) (fears_boils_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_boils_curiosity)) (craves_boils_hamburger)))
 (:action overcome__hangover__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (not (v0_4)) (craves_hangover_guava) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_hangover_entertainment) (not (craves_hangover_guava))))
 (:action overcome__hangover__entertainment__apple
  :parameters ()
  :precondition (and (craves_hangover_apple) (craves_entertainment_apple) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_hangover_entertainment) (not (craves_hangover_apple))))
 (:action overcome__hangover__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (craves_hangover_flounder) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_hangover_flounder)) (fears_hangover_entertainment)))
 (:action overcome__hangover__entertainment__hotdog
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_hotdog) (craves_hangover_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_hangover_entertainment) (not (craves_hangover_hotdog))))
 (:action succumb__boils__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_boils_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_boils_hotdog) (not (fears_boils_intoxication))))
 (:action succumb__boils__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_boils_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_wurst)))
 (:action overcome__laceration__excitement__apple
  :parameters ()
  :precondition (and (not (v4_4)) (craves_laceration_apple) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_laceration_apple)) (fears_laceration_excitement)))
 (:action overcome__laceration__excitement__flounder
  :parameters ()
  :precondition (and (craves_laceration_flounder) (craves_excitement_flounder) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_laceration_flounder)) (fears_laceration_excitement)))
 (:action overcome__laceration__excitement__haroset
  :parameters ()
  :precondition (and (craves_laceration_haroset) (not (v4_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_laceration_haroset)) (fears_laceration_excitement)))
 (:action overcome__laceration__excitement__hamburger
  :parameters ()
  :precondition (and (craves_laceration_hamburger) (not (v4_4)) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_laceration_hamburger)) (fears_laceration_excitement)))
 (:action overcome__laceration__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (not (v4_4)) (craves_laceration_wurst) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_laceration_excitement) (not (craves_laceration_wurst))))
 (:action overcome__laceration__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (craves_laceration_hotdog) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_laceration_excitement) (not (craves_laceration_hotdog))))
 (:action overcome__laceration__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_guava) (craves_laceration_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_laceration_excitement) (not (craves_laceration_guava))))
 (:action succumb__boils__entertainment__wurst
  :parameters ()
  :precondition (and (fears_boils_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_boils_wurst) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (fears_boils_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_boils_hamburger) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (fears_boils_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_boils_guava) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__hotdog
  :parameters ()
  :precondition (and (fears_boils_entertainment) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_boils_hotdog) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (fears_boils_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_boils_apple) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (fears_boils_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_boils_haroset) (not (fears_boils_entertainment))))
 (:action succumb__boils__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_boils_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_boils_flounder) (not (fears_boils_entertainment))))
 (:action overcome__hangover__curiosity__apple
  :parameters ()
  :precondition (and (craves_hangover_apple) (not (v10_4)) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_hangover_curiosity) (not (craves_hangover_apple))))
 (:action overcome__hangover__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_hangover_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_hangover_flounder)) (fears_hangover_curiosity)))
 (:action overcome__hangover__curiosity__haroset
  :parameters ()
  :precondition (and (not (v10_4)) (craves_hangover_haroset) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_hangover_haroset)) (fears_hangover_curiosity)))
 (:action overcome__hangover__curiosity__hamburger
  :parameters ()
  :precondition (and (not (v10_4)) (craves_hangover_hamburger) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_hangover_hamburger)) (fears_hangover_curiosity)))
 (:action overcome__hangover__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_hangover_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_hangover_hotdog)) (fears_hangover_curiosity)))
 (:action succumb__angina__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_angina_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_angina_wurst) (not (fears_angina_excitement))))
 (:action feast__entertainment__guava__apple
  :parameters ()
  :precondition (and (not (v7_4)) (craves_entertainment_guava) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (craves_entertainment_apple) (not (craves_entertainment_guava))))
 (:action succumb__angina__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_angina_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_angina_flounder) (not (fears_angina_excitement))))
 (:action feast__entertainment__guava__haroset
  :parameters ()
  :precondition (and (not (v7_4)) (craves_entertainment_guava) (not (of)))
  :effect (and (when (z_12_0) (v7_0)) (when (not (z_12_0)) (not (v7_0))) (when (z_12_1) (v7_1)) (when (not (z_12_1)) (not (v7_1))) (when (z_12_2) (v7_2)) (when (not (z_12_2)) (not (v7_2))) (when (z_12_3) (v7_3)) (when (not (z_12_3)) (not (v7_3))) (when (z_12_4) (v7_4)) (when (not (z_12_4)) (not (v7_4))) (when (and (v7_4) (not (z_12_4))) (of)) (craves_entertainment_haroset) (not (craves_entertainment_guava))))
 (:action overcome__hangover__intoxication__apple
  :parameters ()
  :precondition (and (craves_hangover_apple) (craves_intoxication_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_apple))))
 (:action overcome__hangover__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (craves_hangover_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_hangover_flounder)) (fears_hangover_intoxication)))
 (:action overcome__hangover__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_hangover_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_hotdog))))
 (:action succumb__angina__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_angina_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_wurst)))
 (:action succumb__angina__curiosity__hamburger
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_hamburger)))
 (:action succumb__angina__curiosity__guava
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_angina_guava) (not (fears_angina_curiosity))))
 (:action succumb__angina__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_angina_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_angina_curiosity)) (craves_angina_hotdog)))
 (:action overcome__laceration__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_laceration_hotdog) (not (v0_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_laceration_hotdog)) (fears_laceration_entertainment)))
 (:action overcome__laceration__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (not (v0_4)) (craves_laceration_guava) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_laceration_guava)) (fears_laceration_entertainment)))
 (:action overcome__laceration__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (not (v0_4)) (craves_laceration_apple) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_laceration_apple)) (fears_laceration_entertainment)))
 (:action overcome__laceration__entertainment__flounder
  :parameters ()
  :precondition (and (craves_laceration_flounder) (craves_entertainment_flounder) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_laceration_flounder)) (fears_laceration_entertainment)))
 (:action overcome__laceration__entertainment__haroset
  :parameters ()
  :precondition (and (craves_laceration_haroset) (craves_entertainment_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_laceration_haroset)) (fears_laceration_entertainment)))
 (:action overcome__laceration__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_laceration_hamburger) (craves_entertainment_hamburger) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_laceration_hamburger)) (fears_laceration_entertainment)))
 (:action overcome__jealousy__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (craves_jealousy_hotdog) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_jealousy_excitement) (not (craves_jealousy_hotdog))))
 (:action overcome__jealousy__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (not (v4_4)) (craves_jealousy_flounder) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_jealousy_flounder)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__excitement__hamburger
  :parameters ()
  :precondition (and (craves_jealousy_hamburger) (not (v4_4)) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_jealousy_hamburger)) (fears_jealousy_excitement)))
 (:action overcome__laceration__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_laceration_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_laceration_curiosity) (not (craves_laceration_guava))))
 (:action overcome__laceration__curiosity__apple
  :parameters ()
  :precondition (and (not (v10_4)) (craves_curiosity_apple) (craves_laceration_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_laceration_apple)) (fears_laceration_curiosity)))
 (:action overcome__laceration__curiosity__flounder
  :parameters ()
  :precondition (and (craves_laceration_flounder) (craves_curiosity_flounder) (not (v10_4)) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_laceration_flounder)) (fears_laceration_curiosity)))
 (:action overcome__laceration__curiosity__haroset
  :parameters ()
  :precondition (and (craves_laceration_haroset) (not (v10_4)) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_laceration_curiosity) (not (craves_laceration_haroset))))
 (:action overcome__laceration__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_laceration_hamburger) (not (v10_4)) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_laceration_hamburger)) (fears_laceration_curiosity)))
 (:action overcome__laceration__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_laceration_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_laceration_curiosity) (not (craves_laceration_hotdog))))
 (:action succumb__sciatica__excitement__guava
  :parameters ()
  :precondition (and (craves_excitement_guava) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_sciatica_guava) (not (fears_sciatica_excitement))))
 (:action overcome__laceration__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_laceration_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_hotdog))))
 (:action overcome__laceration__intoxication__guava
  :parameters ()
  :precondition (and (craves_laceration_guava) (craves_intoxication_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_guava))))
 (:action succumb__sciatica__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_sciatica_excitement)) (craves_sciatica_flounder)))
 (:action succumb__sciatica__excitement__apple
  :parameters ()
  :precondition (and (fears_sciatica_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_sciatica_excitement)) (craves_sciatica_apple)))
 (:action succumb__sciatica__excitement__hamburger
  :parameters ()
  :precondition (and (fears_sciatica_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_sciatica_hamburger) (not (fears_sciatica_excitement))))
 (:action succumb__sciatica__excitement__haroset
  :parameters ()
  :precondition (and (fears_sciatica_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_sciatica_haroset) (not (fears_sciatica_excitement))))
 (:action succumb__sciatica__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_sciatica_excitement)) (craves_sciatica_hotdog)))
 (:action succumb__sciatica__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_sciatica_excitement)) (craves_sciatica_wurst)))
 (:action overcome__laceration__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (not (v1_4)) (craves_laceration_apple) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_apple))))
 (:action overcome__laceration__intoxication__flounder
  :parameters ()
  :precondition (and (craves_laceration_flounder) (craves_intoxication_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_laceration_flounder)) (fears_laceration_intoxication)))
 (:action overcome__laceration__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (craves_laceration_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_haroset))))
 (:action overcome__laceration__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_laceration_hamburger) (craves_intoxication_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_laceration_hamburger)) (fears_laceration_intoxication)))
 (:action overcome__laceration__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_laceration_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_wurst))))
 (:action overcome__jealousy__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_jealousy_hamburger) (craves_entertainment_hamburger) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_jealousy_hamburger)) (fears_jealousy_entertainment)))
 (:action overcome__jealousy__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_jealousy_hotdog) (not (v0_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_jealousy_entertainment) (not (craves_jealousy_hotdog))))
 (:action overcome__jealousy__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (not (v0_4)) (craves_jealousy_guava) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_jealousy_guava)) (fears_jealousy_entertainment)))
 (:action overcome__jealousy__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (not (v0_4)) (craves_jealousy_apple) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_jealousy_entertainment) (not (craves_jealousy_apple))))
 (:action overcome__jealousy__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_jealousy_flounder) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_jealousy_flounder)) (fears_jealousy_entertainment)))
 (:action overcome__grief__excitement__hamburger
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_hamburger) (craves_grief_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_grief_hamburger)) (fears_grief_excitement)))
 (:action overcome__grief__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (not (v4_4)) (craves_grief_hotdog) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_grief_hotdog)) (fears_grief_excitement)))
 (:action overcome__grief__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (not (v4_4)) (craves_grief_flounder) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_grief_flounder)) (fears_grief_excitement)))
 (:action overcome__jealousy__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_jealousy_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_jealousy_curiosity) (not (craves_jealousy_hotdog))))
 (:action overcome__jealousy__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_jealousy_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_jealousy_curiosity) (not (craves_jealousy_flounder))))
 (:action overcome__jealousy__curiosity__haroset
  :parameters ()
  :precondition (and (craves_jealousy_haroset) (not (v10_4)) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_jealousy_curiosity) (not (craves_jealousy_haroset))))
 (:action overcome__jealousy__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_jealousy_hamburger) (not (v10_4)) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_jealousy_hamburger)) (fears_jealousy_curiosity)))
 (:action overcome__jealousy__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_jealousy_hamburger) (craves_intoxication_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_jealousy_hamburger)) (fears_jealousy_intoxication)))
 (:action overcome__jealousy__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_jealousy_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_jealousy_hotdog)) (fears_jealousy_intoxication)))
 (:action overcome__jealousy__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (not (v1_4)) (craves_jealousy_flounder) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_jealousy_flounder)) (fears_jealousy_intoxication)))
 (:action succumb__prostatitis__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_prostatitis_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_prostatitis_wurst) (not (fears_prostatitis_curiosity))))
 (:action succumb__prostatitis__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_curiosity_hamburger) (fears_prostatitis_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_prostatitis_hamburger) (not (fears_prostatitis_curiosity))))
 (:action succumb__prostatitis__curiosity__guava
  :parameters ()
  :precondition (and (fears_prostatitis_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_prostatitis_guava) (not (fears_prostatitis_curiosity))))
 (:action succumb__prostatitis__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_prostatitis_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_prostatitis_hotdog) (not (fears_prostatitis_curiosity))))
 (:action overcome__grief__entertainment__apple
  :parameters ()
  :precondition (and (craves_grief_apple) (craves_entertainment_apple) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_grief_apple)) (fears_grief_entertainment)))
 (:action overcome__grief__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_grief_flounder) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_grief_flounder)) (fears_grief_entertainment)))
 (:action overcome__grief__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (not (v0_4)) (craves_grief_hamburger) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_grief_hamburger)) (fears_grief_entertainment)))
 (:action succumb__prostatitis__curiosity__apple
  :parameters ()
  :precondition (and (fears_prostatitis_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_prostatitis_apple) (not (fears_prostatitis_curiosity))))
 (:action overcome__grief__entertainment__hotdog
  :parameters ()
  :precondition (and (not (v0_4)) (craves_grief_hotdog) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_grief_hotdog)) (fears_grief_entertainment)))
 (:action overcome__grief__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (not (v0_4)) (craves_grief_guava) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_grief_guava)) (fears_grief_entertainment)))
 (:action succumb__prostatitis__curiosity__haroset
  :parameters ()
  :precondition (and (fears_prostatitis_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_prostatitis_haroset) (not (fears_prostatitis_curiosity))))
 (:action succumb__prostatitis__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (fears_prostatitis_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_prostatitis_curiosity)) (craves_prostatitis_flounder)))
 (:action succumb__prostatitis__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_prostatitis_hotdog) (not (fears_prostatitis_intoxication))))
 (:action succumb__prostatitis__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_prostatitis_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_prostatitis_wurst) (not (fears_prostatitis_intoxication))))
 (:action succumb__prostatitis__entertainment__haroset
  :parameters ()
  :precondition (and (fears_prostatitis_entertainment) (craves_entertainment_haroset) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_prostatitis_haroset) (not (fears_prostatitis_entertainment))))
 (:action succumb__prostatitis__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_prostatitis_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_prostatitis_entertainment)) (craves_prostatitis_flounder)))
 (:action succumb__prostatitis__entertainment__wurst
  :parameters ()
  :precondition (and (fears_prostatitis_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_prostatitis_wurst) (not (fears_prostatitis_entertainment))))
 (:action succumb__prostatitis__entertainment__hamburger
  :parameters ()
  :precondition (and (fears_prostatitis_entertainment) (craves_entertainment_hamburger) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_prostatitis_hamburger) (not (fears_prostatitis_entertainment))))
 (:action succumb__prostatitis__entertainment__guava
  :parameters ()
  :precondition (and (fears_prostatitis_entertainment) (craves_entertainment_guava) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_prostatitis_guava) (not (fears_prostatitis_entertainment))))
 (:action succumb__prostatitis__entertainment__hotdog
  :parameters ()
  :precondition (and (fears_prostatitis_entertainment) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_prostatitis_hotdog) (not (fears_prostatitis_entertainment))))
 (:action overcome__grief__curiosity__haroset
  :parameters ()
  :precondition (and (not (v10_4)) (craves_grief_haroset) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_grief_curiosity) (not (craves_grief_haroset))))
 (:action overcome__grief__curiosity__hamburger
  :parameters ()
  :precondition (and (not (v10_4)) (craves_curiosity_hamburger) (craves_grief_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_grief_hamburger)) (fears_grief_curiosity)))
 (:action overcome__grief__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_grief_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_grief_hotdog)) (fears_grief_curiosity)))
 (:action succumb__prostatitis__entertainment__apple
  :parameters ()
  :precondition (and (fears_prostatitis_entertainment) (craves_entertainment_apple) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_prostatitis_apple) (not (fears_prostatitis_entertainment))))
 (:action overcome__grief__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_grief_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_grief_flounder)) (fears_grief_curiosity)))
 (:action succumb__anger__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_anger_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anger_wurst) (not (fears_anger_excitement))))
 (:action overcome__grief__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (craves_grief_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_grief_flounder)) (fears_grief_intoxication)))
 (:action overcome__grief__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (not (v1_4)) (craves_grief_hamburger) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_grief_hamburger)) (fears_grief_intoxication)))
 (:action overcome__grief__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_grief_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_grief_hotdog)) (fears_grief_intoxication)))
 (:action succumb__anger__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_anger_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anger_flounder) (not (fears_anger_excitement))))
 (:action succumb__sciatica__curiosity__guava
  :parameters ()
  :precondition (and (fears_sciatica_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_guava)))
 (:action succumb__sciatica__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_sciatica_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_hotdog)))
 (:action succumb__sciatica__curiosity__apple
  :parameters ()
  :precondition (and (craves_curiosity_apple) (fears_sciatica_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_apple)))
 (:action succumb__sciatica__curiosity__haroset
  :parameters ()
  :precondition (and (craves_curiosity_haroset) (fears_sciatica_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_sciatica_haroset) (not (fears_sciatica_curiosity))))
 (:action succumb__sciatica__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (fears_sciatica_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_flounder)))
 (:action succumb__sciatica__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_sciatica_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_wurst)))
 (:action succumb__sciatica__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_curiosity_hamburger) (fears_sciatica_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_sciatica_curiosity)) (craves_sciatica_hamburger)))
 (:action succumb__sciatica__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_sciatica_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_sciatica_intoxication)) (craves_sciatica_hotdog)))
 (:action succumb__sciatica__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_sciatica_intoxication)) (craves_sciatica_wurst)))
 (:action succumb__sciatica__intoxication__guava
  :parameters ()
  :precondition (and (fears_sciatica_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_sciatica_guava) (not (fears_sciatica_intoxication))))
 (:action succumb__sciatica__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_sciatica_flounder) (not (fears_sciatica_intoxication))))
 (:action succumb__sciatica__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_sciatica_intoxication)) (craves_sciatica_apple)))
 (:action succumb__sciatica__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_sciatica_hamburger) (not (fears_sciatica_intoxication))))
 (:action succumb__sciatica__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_sciatica_haroset) (not (fears_sciatica_intoxication))))
 (:action succumb__sciatica__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (fears_sciatica_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_sciatica_guava) (not (fears_sciatica_entertainment))))
 (:action succumb__sciatica__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (fears_sciatica_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_sciatica_entertainment)) (craves_sciatica_apple)))
 (:action succumb__prostatitis__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_prostatitis_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_prostatitis_wurst) (not (fears_prostatitis_excitement))))
 (:action succumb__prostatitis__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_prostatitis_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_prostatitis_excitement)) (craves_prostatitis_flounder)))
 (:action succumb__hangover__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_hangover_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_hangover_entertainment)) (craves_hangover_flounder)))
 (:action succumb__hangover__entertainment__apple
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_apple) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_hangover_entertainment)) (craves_hangover_apple)))
 (:action succumb__hangover__entertainment__hamburger
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_hamburger) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_hangover_hamburger) (not (fears_hangover_entertainment))))
 (:action succumb__hangover__entertainment__haroset
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_haroset) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_hangover_haroset) (not (fears_hangover_entertainment))))
 (:action succumb__hangover__entertainment__hotdog
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_hangover_hotdog) (not (fears_hangover_entertainment))))
 (:action succumb__hangover__entertainment__wurst
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_hangover_entertainment)) (craves_hangover_wurst)))
 (:action overcome__sciatica__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_sciatica_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_hotdog))))
 (:action overcome__sciatica__curiosity__haroset
  :parameters ()
  :precondition (and (craves_sciatica_haroset) (not (v10_4)) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_haroset))))
 (:action overcome__sciatica__curiosity__hamburger
  :parameters ()
  :precondition (and (not (v10_4)) (craves_sciatica_hamburger) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_hamburger))))
 (:action overcome__sciatica__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_sciatica_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_sciatica_wurst)) (fears_sciatica_intoxication)))
 (:action overcome__sciatica__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_sciatica_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_sciatica_intoxication) (not (craves_sciatica_hotdog))))
 (:action overcome__sciatica__intoxication__haroset
  :parameters ()
  :precondition (and (craves_sciatica_haroset) (craves_intoxication_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_sciatica_haroset)) (fears_sciatica_intoxication)))
 (:action succumb__laceration__curiosity__hamburger
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_laceration_hamburger) (not (fears_laceration_curiosity))))
 (:action succumb__laceration__curiosity__haroset
  :parameters ()
  :precondition (and (fears_laceration_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_laceration_haroset) (not (fears_laceration_curiosity))))
 (:action feast__excitement__flounder__wurst
  :parameters ()
  :precondition (and (not (v3_4)) (craves_excitement_flounder) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (craves_excitement_wurst) (not (craves_excitement_flounder))))
 (:action feast__excitement__flounder__hamburger
  :parameters ()
  :precondition (and (not (v3_4)) (craves_excitement_flounder) (not (of)))
  :effect (and (when (z_7_0) (v3_0)) (when (not (z_7_0)) (not (v3_0))) (when (z_7_1) (v3_1)) (when (not (z_7_1)) (not (v3_1))) (when (z_7_2) (v3_2)) (when (not (z_7_2)) (not (v3_2))) (when (z_7_3) (v3_3)) (when (not (z_7_3)) (not (v3_3))) (when (z_7_4) (v3_4)) (when (not (z_7_4)) (not (v3_4))) (when (and (v3_4) (not (z_7_4))) (of)) (not (craves_excitement_flounder)) (craves_excitement_hamburger)))
 (:action overcome__sciatica__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (not (v4_4)) (craves_sciatica_flounder) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_flounder))))
 (:action succumb__laceration__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (fears_laceration_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_laceration_guava) (not (fears_laceration_entertainment))))
 (:action overcome__sciatica__excitement__haroset
  :parameters ()
  :precondition (and (craves_sciatica_haroset) (not (v4_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_sciatica_haroset)) (fears_sciatica_excitement)))
 (:action overcome__sciatica__excitement__wurst
  :parameters ()
  :precondition (and (craves_sciatica_wurst) (craves_excitement_wurst) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_sciatica_wurst)) (fears_sciatica_excitement)))
 (:action overcome__sciatica__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (craves_sciatica_hotdog) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_hotdog))))
 (:action succumb__loneliness__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_loneliness_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_loneliness_flounder) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__apple
  :parameters ()
  :precondition (and (fears_loneliness_entertainment) (craves_entertainment_apple) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_loneliness_entertainment)) (craves_loneliness_apple)))
 (:action succumb__loneliness__entertainment__hamburger
  :parameters ()
  :precondition (and (fears_loneliness_entertainment) (craves_entertainment_hamburger) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_loneliness_hamburger) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__haroset
  :parameters ()
  :precondition (and (fears_loneliness_entertainment) (craves_entertainment_haroset) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_loneliness_haroset) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__hotdog
  :parameters ()
  :precondition (and (fears_loneliness_entertainment) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_loneliness_hotdog) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__wurst
  :parameters ()
  :precondition (and (fears_loneliness_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_loneliness_wurst) (not (fears_loneliness_entertainment))))
 (:action succumb__loneliness__entertainment__guava
  :parameters ()
  :precondition (and (fears_loneliness_entertainment) (craves_entertainment_guava) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_loneliness_guava) (not (fears_loneliness_entertainment))))
 (:action overcome__prostatitis__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_prostatitis_hamburger) (not (v10_4)) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_prostatitis_hamburger)) (fears_prostatitis_curiosity)))
 (:action overcome__prostatitis__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_prostatitis_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_prostatitis_hotdog)) (fears_prostatitis_curiosity)))
 (:action overcome__prostatitis__curiosity__apple
  :parameters ()
  :precondition (and (not (v10_4)) (craves_prostatitis_apple) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_prostatitis_apple)) (fears_prostatitis_curiosity)))
 (:action overcome__prostatitis__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_prostatitis_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_prostatitis_flounder)) (fears_prostatitis_curiosity)))
 (:action overcome__prostatitis__curiosity__haroset
  :parameters ()
  :precondition (and (craves_prostatitis_haroset) (not (v10_4)) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_prostatitis_haroset)) (fears_prostatitis_curiosity)))
 (:action succumb__hangover__excitement__flounder
  :parameters ()
  :precondition (and (fears_hangover_excitement) (craves_excitement_flounder) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_hangover_excitement)) (craves_hangover_flounder)))
 (:action succumb__hangover__excitement__wurst
  :parameters ()
  :precondition (and (fears_hangover_excitement) (craves_excitement_wurst) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_hangover_excitement)) (craves_hangover_wurst)))
 (:action overcome__prostatitis__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_prostatitis_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_prostatitis_intoxication) (not (craves_prostatitis_hotdog))))
 (:action overcome__prostatitis__intoxication__apple
  :parameters ()
  :precondition (and (craves_prostatitis_apple) (craves_intoxication_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_prostatitis_apple)) (fears_prostatitis_intoxication)))
 (:action overcome__prostatitis__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (craves_prostatitis_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_prostatitis_flounder)) (fears_prostatitis_intoxication)))
 (:action succumb__hangover__curiosity__flounder
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_hangover_curiosity)) (craves_hangover_flounder)))
 (:action succumb__hangover__curiosity__apple
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_hangover_apple) (not (fears_hangover_curiosity))))
 (:action succumb__hangover__curiosity__hamburger
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_hangover_hamburger) (not (fears_hangover_curiosity))))
 (:action succumb__hangover__curiosity__haroset
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_hangover_haroset) (not (fears_hangover_curiosity))))
 (:action succumb__hangover__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_hangover_hotdog) (not (fears_hangover_curiosity))))
 (:action succumb__hangover__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_hangover_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_hangover_curiosity)) (craves_hangover_wurst)))
 (:action succumb__hangover__curiosity__guava
  :parameters ()
  :precondition (and (fears_hangover_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_hangover_guava) (not (fears_hangover_curiosity))))
 (:action overcome__sciatica__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_sciatica_hotdog) (not (v0_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_sciatica_entertainment) (not (craves_sciatica_hotdog))))
 (:action overcome__sciatica__entertainment__haroset
  :parameters ()
  :precondition (and (craves_sciatica_haroset) (craves_entertainment_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_sciatica_haroset)) (fears_sciatica_entertainment)))
 (:action succumb__hangover__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_hangover_intoxication)) (craves_hangover_wurst)))
 (:action succumb__hangover__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_hangover_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_hangover_hotdog) (not (fears_hangover_intoxication))))
 (:action feast__excitement__wurst__flounder
  :parameters ()
  :precondition (and (craves_excitement_wurst) (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (not (craves_excitement_wurst)) (craves_excitement_flounder)))
 (:action feast__excitement__wurst__hotdog
  :parameters ()
  :precondition (and (craves_excitement_wurst) (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (not (craves_excitement_wurst)) (craves_excitement_hotdog)))
 (:action overcome__prostatitis__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (not (v4_4)) (craves_prostatitis_hotdog) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_prostatitis_hotdog)) (fears_prostatitis_excitement)))
 (:action succumb__hangover__entertainment__guava
  :parameters ()
  :precondition (and (fears_hangover_entertainment) (craves_entertainment_guava) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_hangover_entertainment)) (craves_hangover_guava)))
 (:action overcome__prostatitis__excitement__apple
  :parameters ()
  :precondition (and (craves_prostatitis_apple) (not (v4_4)) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_prostatitis_apple)) (fears_prostatitis_excitement)))
 (:action overcome__prostatitis__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (not (v4_4)) (craves_prostatitis_flounder) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_prostatitis_flounder)) (fears_prostatitis_excitement)))
 (:action succumb__jealousy__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_jealousy_flounder) (not (fears_jealousy_entertainment))))
 (:action succumb__jealousy__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_apple)))
 (:action overcome__anger__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_anger_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_anger_flounder)) (fears_anger_curiosity)))
 (:action overcome__anger__curiosity__haroset
  :parameters ()
  :precondition (and (not (v10_4)) (craves_curiosity_haroset) (craves_anger_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_anger_haroset)) (fears_anger_curiosity)))
 (:action overcome__anger__curiosity__hamburger
  :parameters ()
  :precondition (and (not (v10_4)) (craves_anger_hamburger) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_anger_curiosity) (not (craves_anger_hamburger))))
 (:action overcome__anger__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_anger_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_anger_hotdog)) (fears_anger_curiosity)))
 (:action succumb__grief__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_grief_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_grief_excitement)) (craves_grief_wurst)))
 (:action overcome__anger__curiosity__apple
  :parameters ()
  :precondition (and (not (v10_4)) (craves_anger_apple) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_anger_curiosity) (not (craves_anger_apple))))
 (:action succumb__grief__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_grief_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_grief_flounder) (not (fears_grief_excitement))))
 (:action overcome__anger__intoxication__apple
  :parameters ()
  :precondition (and (craves_anger_apple) (craves_intoxication_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_anger_intoxication) (not (craves_anger_apple))))
 (:action overcome__anger__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (craves_anger_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_anger_intoxication) (not (craves_anger_flounder))))
 (:action overcome__anger__intoxication__hotdog
  :parameters ()
  :precondition (and (not (v1_4)) (craves_anger_hotdog) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_anger_hotdog)) (fears_anger_intoxication)))
 (:action succumb__grief__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_curiosity_hamburger) (fears_grief_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_grief_curiosity)) (craves_grief_hamburger)))
 (:action succumb__grief__curiosity__haroset
  :parameters ()
  :precondition (and (craves_curiosity_haroset) (fears_grief_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_grief_haroset) (not (fears_grief_curiosity))))
 (:action succumb__grief__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_grief_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_grief_hotdog) (not (fears_grief_curiosity))))
 (:action succumb__grief__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_grief_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_grief_curiosity)) (craves_grief_wurst)))
 (:action succumb__grief__curiosity__guava
  :parameters ()
  :precondition (and (craves_curiosity_guava) (fears_grief_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_grief_guava) (not (fears_grief_curiosity))))
 (:action succumb__grief__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (fears_grief_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_grief_flounder) (not (fears_grief_curiosity))))
 (:action feast__intoxication__hotdog__apple
  :parameters ()
  :precondition (and (not (v2_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (not (craves_intoxication_hotdog)) (craves_intoxication_apple)))
 (:action succumb__grief__curiosity__apple
  :parameters ()
  :precondition (and (craves_curiosity_apple) (fears_grief_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_grief_apple) (not (fears_grief_curiosity))))
 (:action feast__intoxication__hotdog__wurst
  :parameters ()
  :precondition (and (not (v2_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_5_0) (v2_0)) (when (not (z_5_0)) (not (v2_0))) (when (z_5_1) (v2_1)) (when (not (z_5_1)) (not (v2_1))) (when (z_5_2) (v2_2)) (when (not (z_5_2)) (not (v2_2))) (when (z_5_3) (v2_3)) (when (not (z_5_3)) (not (v2_3))) (when (z_5_4) (v2_4)) (when (not (z_5_4)) (not (v2_4))) (when (and (v2_4) (not (z_5_4))) (of)) (craves_intoxication_wurst) (not (craves_intoxication_hotdog))))
 (:action overcome__prostatitis__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_prostatitis_flounder) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_prostatitis_flounder)) (fears_prostatitis_entertainment)))
 (:action overcome__prostatitis__entertainment__hotdog
  :parameters ()
  :precondition (and (not (v0_4)) (craves_prostatitis_hotdog) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_prostatitis_hotdog)) (fears_prostatitis_entertainment)))
 (:action overcome__prostatitis__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (not (v0_4)) (craves_prostatitis_guava) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_prostatitis_guava)) (fears_prostatitis_entertainment)))
 (:action succumb__grief__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_grief_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_grief_intoxication)) (craves_grief_wurst)))
 (:action succumb__grief__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_grief_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_grief_hotdog) (not (fears_grief_intoxication))))
 (:action overcome__prostatitis__entertainment__apple
  :parameters ()
  :precondition (and (craves_prostatitis_apple) (craves_entertainment_apple) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_prostatitis_apple)) (fears_prostatitis_entertainment)))
 (:action feast__intoxication__wurst__flounder
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (not (craves_intoxication_wurst)) (craves_intoxication_flounder)))
 (:action feast__intoxication__wurst__hotdog
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (not (v8_4)) (not (of)))
  :effect (and (when (z_13_0) (v8_0)) (when (not (z_13_0)) (not (v8_0))) (when (z_13_1) (v8_1)) (when (not (z_13_1)) (not (v8_1))) (when (z_13_2) (v8_2)) (when (not (z_13_2)) (not (v8_2))) (when (z_13_3) (v8_3)) (when (not (z_13_3)) (not (v8_3))) (when (z_13_4) (v8_4)) (when (not (z_13_4)) (not (v8_4))) (when (and (v8_4) (not (z_13_4))) (of)) (craves_intoxication_hotdog) (not (craves_intoxication_wurst))))
 (:action overcome__anger__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (not (v4_4)) (craves_anger_hotdog) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anger_hotdog)) (fears_anger_excitement)))
 (:action succumb__grief__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_grief_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_grief_flounder) (not (fears_grief_entertainment))))
 (:action succumb__grief__entertainment__apple
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_apple) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_grief_apple) (not (fears_grief_entertainment))))
 (:action succumb__grief__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (fears_grief_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_grief_entertainment)) (craves_grief_hamburger)))
 (:action succumb__grief__entertainment__haroset
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_haroset) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_grief_haroset) (not (fears_grief_entertainment))))
 (:action succumb__grief__entertainment__hotdog
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_grief_hotdog) (not (fears_grief_entertainment))))
 (:action succumb__grief__entertainment__wurst
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_grief_entertainment)) (craves_grief_wurst)))
 (:action overcome__anger__excitement__apple
  :parameters ()
  :precondition (and (craves_anger_apple) (not (v4_4)) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anger_apple)) (fears_anger_excitement)))
 (:action overcome__anger__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (craves_anger_flounder) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anger_flounder)) (fears_anger_excitement)))
 (:action succumb__grief__entertainment__guava
  :parameters ()
  :precondition (and (fears_grief_entertainment) (craves_entertainment_guava) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_grief_guava) (not (fears_grief_entertainment))))
 (:action succumb__laceration__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (fears_laceration_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_laceration_apple) (not (fears_laceration_entertainment))))
 (:action overcome__anxiety__curiosity__apple
  :parameters ()
  :precondition (and (craves_anxiety_apple) (not (v10_4)) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_anxiety_apple)) (fears_anxiety_curiosity)))
 (:action overcome__anxiety__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (craves_anxiety_flounder) (not (v10_4)) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_anxiety_curiosity) (not (craves_anxiety_flounder))))
 (:action overcome__anxiety__curiosity__haroset
  :parameters ()
  :precondition (and (not (v10_4)) (craves_anxiety_haroset) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_anxiety_curiosity) (not (craves_anxiety_haroset))))
 (:action overcome__anxiety__curiosity__hamburger
  :parameters ()
  :precondition (and (not (v10_4)) (craves_anxiety_hamburger) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_anxiety_curiosity) (not (craves_anxiety_hamburger))))
 (:action overcome__anxiety__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_anxiety_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_anxiety_curiosity) (not (craves_anxiety_hotdog))))
 (:action overcome__anxiety__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_anxiety_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_anxiety_curiosity) (not (craves_anxiety_guava))))
 (:action succumb__jealousy__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_jealousy_flounder) (not (fears_jealousy_excitement))))
 (:action succumb__jealousy__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_wurst)))
 (:action overcome__anxiety__intoxication__guava
  :parameters ()
  :precondition (and (craves_intoxication_guava) (craves_anxiety_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_anxiety_intoxication) (not (craves_anxiety_guava))))
 (:action overcome__anxiety__intoxication__apple
  :parameters ()
  :precondition (and (craves_anxiety_apple) (craves_intoxication_apple) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_anxiety_apple)) (fears_anxiety_intoxication)))
 (:action overcome__anxiety__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (craves_anxiety_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_anxiety_intoxication) (not (craves_anxiety_flounder))))
 (:action overcome__anxiety__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (craves_anxiety_haroset) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_anxiety_intoxication) (not (craves_anxiety_haroset))))
 (:action overcome__anxiety__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_anxiety_hamburger) (craves_intoxication_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_anxiety_intoxication) (not (craves_anxiety_hamburger))))
 (:action overcome__anxiety__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_anxiety_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_anxiety_wurst)) (fears_anxiety_intoxication)))
 (:action overcome__anxiety__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_anxiety_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_anxiety_intoxication) (not (craves_anxiety_hotdog))))
 (:action succumb__jealousy__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_jealousy_curiosity)) (craves_jealousy_hotdog)))
 (:action succumb__jealousy__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_jealousy_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_jealousy_curiosity)) (craves_jealousy_wurst)))
 (:action succumb__jealousy__curiosity__guava
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_jealousy_curiosity)) (craves_jealousy_guava)))
 (:action succumb__jealousy__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (fears_jealousy_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_jealousy_flounder) (not (fears_jealousy_curiosity))))
 (:action succumb__jealousy__curiosity__apple
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_jealousy_curiosity)) (craves_jealousy_apple)))
 (:action succumb__jealousy__curiosity__hamburger
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_jealousy_curiosity)) (craves_jealousy_hamburger)))
 (:action succumb__jealousy__curiosity__haroset
  :parameters ()
  :precondition (and (fears_jealousy_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_jealousy_haroset) (not (fears_jealousy_curiosity))))
 (:action overcome__anger__entertainment__apple
  :parameters ()
  :precondition (and (craves_anger_apple) (craves_entertainment_apple) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anger_entertainment) (not (craves_anger_apple))))
 (:action overcome__anger__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (craves_anger_flounder) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anger_entertainment) (not (craves_anger_flounder))))
 (:action overcome__anger__entertainment__hotdog
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_hotdog) (craves_anger_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anger_entertainment) (not (craves_anger_hotdog))))
 (:action overcome__anger__entertainment__guava
  :parameters ()
  :precondition (and (craves_anger_guava) (craves_entertainment_guava) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anger_entertainment) (not (craves_anger_guava))))
 (:action succumb__jealousy__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_jealousy_wurst) (not (fears_jealousy_intoxication))))
 (:action succumb__jealousy__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_jealousy_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_jealousy_hotdog) (not (fears_jealousy_intoxication))))
 (:action overcome__anxiety__excitement__apple
  :parameters ()
  :precondition (and (craves_anxiety_apple) (not (v4_4)) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anxiety_apple)) (fears_anxiety_excitement)))
 (:action overcome__anxiety__excitement__flounder
  :parameters ()
  :precondition (and (craves_anxiety_flounder) (craves_excitement_flounder) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anxiety_flounder)) (fears_anxiety_excitement)))
 (:action overcome__anxiety__excitement__haroset
  :parameters ()
  :precondition (and (craves_anxiety_haroset) (not (v4_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anxiety_haroset)) (fears_anxiety_excitement)))
 (:action overcome__anxiety__excitement__hamburger
  :parameters ()
  :precondition (and (not (v4_4)) (craves_anxiety_hamburger) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anxiety_hamburger)) (fears_anxiety_excitement)))
 (:action overcome__anxiety__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (not (v4_4)) (craves_anxiety_wurst) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anxiety_wurst)) (fears_anxiety_excitement)))
 (:action overcome__anxiety__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (not (v4_4)) (craves_anxiety_hotdog) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_anxiety_excitement) (not (craves_anxiety_hotdog))))
 (:action overcome__anxiety__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_guava) (craves_anxiety_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anxiety_guava)) (fears_anxiety_excitement)))
 (:action succumb__jealousy__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_hamburger)))
 (:action succumb__jealousy__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_jealousy_haroset) (not (fears_jealousy_entertainment))))
 (:action succumb__jealousy__entertainment__hotdog
  :parameters ()
  :precondition (and (fears_jealousy_entertainment) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_jealousy_hotdog) (not (fears_jealousy_entertainment))))
 (:action succumb__jealousy__entertainment__wurst
  :parameters ()
  :precondition (and (fears_jealousy_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_jealousy_entertainment)) (craves_jealousy_wurst)))
 (:action succumb__jealousy__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (fears_jealousy_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (craves_jealousy_guava) (not (fears_jealousy_entertainment))))
 (:action overcome__boils__curiosity__hotdog
  :parameters ()
  :precondition (and (not (v10_4)) (craves_boils_hotdog) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_boils_hotdog)) (fears_boils_curiosity)))
 (:action overcome__boils__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_boils_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_boils_curiosity) (not (craves_boils_flounder))))
 (:action overcome__boils__curiosity__haroset
  :parameters ()
  :precondition (and (craves_boils_haroset) (not (v10_4)) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_boils_curiosity) (not (craves_boils_haroset))))
 (:action overcome__boils__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_boils_hamburger) (not (v10_4)) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_boils_curiosity) (not (craves_boils_hamburger))))
 (:action succumb__dread__excitement__apple
  :parameters ()
  :precondition (and (fears_dread_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_dread_excitement)) (craves_dread_apple)))
 (:action overcome__boils__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_boils_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_boils_hotdog)) (fears_boils_intoxication)))
 (:action succumb__dread__excitement__haroset
  :parameters ()
  :precondition (and (fears_dread_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_dread_haroset) (not (fears_dread_excitement))))
 (:action succumb__dread__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_dread_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_dread_excitement)) (craves_dread_flounder)))
 (:action succumb__dread__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_dread_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_dread_excitement)) (craves_dread_wurst)))
 (:action succumb__dread__excitement__hamburger
  :parameters ()
  :precondition (and (craves_excitement_hamburger) (fears_dread_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_dread_hamburger) (not (fears_dread_excitement))))
 (:action succumb__dread__excitement__guava
  :parameters ()
  :precondition (and (craves_excitement_guava) (fears_dread_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_dread_excitement)) (craves_dread_guava)))
 (:action succumb__dread__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (fears_dread_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_dread_excitement)) (craves_dread_hotdog)))
 (:action overcome__boils__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (not (v1_4)) (craves_boils_flounder) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_boils_intoxication) (not (craves_boils_flounder))))
 (:action overcome__boils__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_boils_hamburger) (craves_intoxication_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_boils_intoxication) (not (craves_boils_hamburger))))
 (:action succumb__dread__curiosity__guava
  :parameters ()
  :precondition (and (craves_curiosity_guava) (fears_dread_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_dread_curiosity)) (craves_dread_guava)))
 (:action overcome__anxiety__entertainment__hotdog
  :parameters ()
  :precondition (and (not (v0_4)) (craves_anxiety_hotdog) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anxiety_entertainment) (not (craves_anxiety_hotdog))))
 (:action overcome__anxiety__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (not (v0_4)) (craves_anxiety_guava) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anxiety_entertainment) (not (craves_anxiety_guava))))
 (:action succumb__dread__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (fears_dread_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_dread_flounder) (not (fears_dread_curiosity))))
 (:action succumb__dread__curiosity__apple
  :parameters ()
  :precondition (and (craves_curiosity_apple) (fears_dread_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_dread_curiosity)) (craves_dread_apple)))
 (:action succumb__dread__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_curiosity_hamburger) (fears_dread_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_dread_hamburger) (not (fears_dread_curiosity))))
 (:action succumb__dread__curiosity__haroset
  :parameters ()
  :precondition (and (craves_curiosity_haroset) (fears_dread_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (craves_dread_haroset) (not (fears_dread_curiosity))))
 (:action succumb__dread__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_dread_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_dread_curiosity)) (craves_dread_hotdog)))
 (:action succumb__dread__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_dread_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_dread_curiosity)) (craves_dread_wurst)))
 (:action overcome__anxiety__entertainment__apple
  :parameters ()
  :precondition (and (craves_anxiety_apple) (craves_entertainment_apple) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_anxiety_apple)) (fears_anxiety_entertainment)))
 (:action overcome__anxiety__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (craves_anxiety_flounder) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_anxiety_flounder)) (fears_anxiety_entertainment)))
 (:action overcome__anxiety__entertainment__haroset
  :parameters ()
  :precondition (and (craves_anxiety_haroset) (craves_entertainment_haroset) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anxiety_entertainment) (not (craves_anxiety_haroset))))
 (:action overcome__anxiety__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (craves_anxiety_hamburger) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_anxiety_hamburger)) (fears_anxiety_entertainment)))
 (:action succumb__dread__intoxication__guava
  :parameters ()
  :precondition (and (craves_intoxication_guava) (fears_dread_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_dread_intoxication)) (craves_dread_guava)))
 (:action succumb__dread__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_dread_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_dread_intoxication)) (craves_dread_hotdog)))
 (:action succumb__dread__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (fears_dread_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_dread_intoxication)) (craves_dread_apple)))
 (:action succumb__dread__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_dread_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_dread_haroset) (not (fears_dread_intoxication))))
 (:action succumb__dread__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_dread_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_dread_flounder) (not (fears_dread_intoxication))))
 (:action succumb__dread__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_dread_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_dread_intoxication)) (craves_dread_wurst)))
 (:action succumb__dread__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (fears_dread_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_dread_hamburger) (not (fears_dread_intoxication))))
 (:action overcome__boils__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (not (v4_4)) (craves_boils_flounder) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_boils_excitement) (not (craves_boils_flounder))))
 (:action overcome__boils__excitement__hamburger
  :parameters ()
  :precondition (and (craves_boils_hamburger) (not (v4_4)) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_boils_excitement) (not (craves_boils_hamburger))))
 (:action overcome__boils__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (craves_boils_hotdog) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_boils_excitement) (not (craves_boils_hotdog))))
 (:action overcome__angina__curiosity__hamburger
  :parameters ()
  :precondition (and (not (v10_4)) (craves_curiosity_hamburger) (craves_angina_hamburger) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_angina_hamburger)) (fears_angina_curiosity)))
 (:action overcome__angina__curiosity__hotdog
  :parameters ()
  :precondition (and (craves_angina_hotdog) (not (v10_4)) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_angina_curiosity) (not (craves_angina_hotdog))))
 (:action overcome__angina__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (not (v10_4)) (craves_angina_flounder) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_angina_curiosity) (not (craves_angina_flounder))))
 (:action overcome__angina__curiosity__haroset
  :parameters ()
  :precondition (and (not (v10_4)) (craves_angina_haroset) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_angina_curiosity) (not (craves_angina_haroset))))
 (:action overcome__angina__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (craves_angina_hamburger) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_angina_hamburger)) (fears_angina_intoxication)))
 (:action overcome__angina__intoxication__hotdog
  :parameters ()
  :precondition (and (craves_angina_hotdog) (not (v1_4)) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_angina_hotdog)) (fears_angina_intoxication)))
 (:action overcome__angina__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (craves_angina_flounder) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_angina_flounder)) (fears_angina_intoxication)))
 (:action overcome__boils__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_boils_hamburger) (craves_entertainment_hamburger) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_boils_entertainment) (not (craves_boils_hamburger))))
 (:action overcome__boils__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_boils_hotdog) (not (v0_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_boils_hotdog)) (fears_boils_entertainment)))
 (:action overcome__boils__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (not (v0_4)) (craves_boils_guava) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_boils_entertainment) (not (craves_boils_guava))))
 (:action overcome__boils__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (not (v0_4)) (craves_boils_apple) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_boils_apple)) (fears_boils_entertainment)))
 (:action overcome__boils__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (not (v0_4)) (craves_boils_flounder) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_boils_entertainment) (not (craves_boils_flounder))))
 (:action overcome__angina__excitement__hotdog
  :parameters ()
  :precondition (and (craves_angina_hotdog) (craves_excitement_hotdog) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_angina_excitement) (not (craves_angina_hotdog))))
 (:action overcome__angina__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (craves_angina_flounder) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_angina_excitement) (not (craves_angina_flounder))))
 (:action overcome__angina__excitement__hamburger
  :parameters ()
  :precondition (and (not (v4_4)) (craves_angina_hamburger) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_angina_hamburger)) (fears_angina_excitement)))
 (:action succumb__abrasion__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (fears_abrasion_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_hamburger)))
 (:action succumb__abrasion__entertainment__haroset
  :parameters ()
  :precondition (and (craves_entertainment_haroset) (fears_abrasion_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_haroset)))
 (:action succumb__abrasion__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_entertainment_hotdog) (fears_abrasion_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_hotdog)))
 (:action succumb__abrasion__entertainment__wurst
  :parameters ()
  :precondition (and (fears_abrasion_entertainment) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_wurst)))
 (:action succumb__abrasion__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (fears_abrasion_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_guava)))
 (:action succumb__abrasion__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (fears_abrasion_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_flounder)))
 (:action succumb__abrasion__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (fears_abrasion_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_abrasion_entertainment)) (craves_abrasion_apple)))
 (:action succumb__loneliness__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_loneliness_wurst) (not (fears_loneliness_excitement))))
 (:action succumb__loneliness__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_loneliness_flounder) (not (fears_loneliness_excitement))))
 (:action succumb__loneliness__curiosity__hamburger
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_hamburger) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_hamburger)))
 (:action succumb__loneliness__curiosity__haroset
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_haroset)))
 (:action succumb__loneliness__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_hotdog)))
 (:action succumb__loneliness__curiosity__wurst
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_wurst) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_wurst)))
 (:action succumb__loneliness__curiosity__guava
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_guava)))
 (:action overcome__angina__entertainment__flounder
  :parameters ()
  :precondition (and (craves_entertainment_flounder) (craves_angina_flounder) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_angina_entertainment) (not (craves_angina_flounder))))
 (:action overcome__angina__entertainment__hamburger
  :parameters ()
  :precondition (and (craves_entertainment_hamburger) (not (v0_4)) (craves_angina_hamburger) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_angina_hamburger)) (fears_angina_entertainment)))
 (:action overcome__angina__entertainment__hotdog
  :parameters ()
  :precondition (and (craves_angina_hotdog) (not (v0_4)) (craves_entertainment_hotdog) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_angina_entertainment) (not (craves_angina_hotdog))))
 (:action overcome__angina__entertainment__guava
  :parameters ()
  :precondition (and (craves_angina_guava) (craves_entertainment_guava) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_angina_entertainment) (not (craves_angina_guava))))
 (:action succumb__loneliness__curiosity__flounder
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_flounder) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_flounder)))
 (:action succumb__loneliness__curiosity__apple
  :parameters ()
  :precondition (and (fears_loneliness_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_loneliness_curiosity)) (craves_loneliness_apple)))
 (:action overcome__angina__entertainment__apple
  :parameters ()
  :precondition (and (craves_angina_apple) (craves_entertainment_apple) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_angina_entertainment) (not (craves_angina_apple))))
 (:action succumb__loneliness__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_loneliness_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_wurst)))
 (:action succumb__loneliness__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_hotdog)))
 (:action succumb__dread__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_dread_entertainment)) (craves_dread_guava)))
 (:action succumb__dread__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (fears_dread_entertainment) (not (of)))
  :effect (and (when (z_16_0) (v0_0)) (when (not (z_16_0)) (not (v0_0))) (when (z_16_1) (v0_1)) (when (not (z_16_1)) (not (v0_1))) (when (z_16_2) (v0_2)) (when (not (z_16_2)) (not (v0_2))) (when (z_16_3) (v0_3)) (when (not (z_16_3)) (not (v0_3))) (when (z_16_4) (v0_4)) (when (not (z_16_4)) (not (v0_4))) (when (and (not (v0_4)) (z_16_4)) (of)) (not (fears_dread_entertainment)) (craves_dread_apple)))
 (:action succumb__abrasion__excitement__flounder
  :parameters ()
  :precondition (and (craves_excitement_flounder) (fears_abrasion_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_abrasion_flounder) (not (fears_abrasion_excitement))))
 (:action succumb__abrasion__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (fears_abrasion_excitement) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_abrasion_wurst) (not (fears_abrasion_excitement))))
 (:action overcome__dread__excitement__flounder
  :parameters ()
  :precondition (and (craves_dread_flounder) (craves_excitement_flounder) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_dread_excitement) (not (craves_dread_flounder))))
 (:action overcome__dread__excitement__haroset
  :parameters ()
  :precondition (and (craves_dread_haroset) (not (v4_4)) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_dread_haroset)) (fears_dread_excitement)))
 (:action overcome__dread__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (craves_dread_wurst) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_dread_excitement) (not (craves_dread_wurst))))
 (:action overcome__dread__excitement__hotdog
  :parameters ()
  :precondition (and (craves_excitement_hotdog) (not (v4_4)) (craves_dread_hotdog) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_dread_excitement) (not (craves_dread_hotdog))))
 (:action succumb__abrasion__curiosity__hotdog
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_hotdog) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_hotdog)))
 (:action succumb__abrasion__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (fears_abrasion_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_wurst)))
 (:action succumb__abrasion__curiosity__guava
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_guava)))
 (:action succumb__abrasion__curiosity__flounder
  :parameters ()
  :precondition (and (craves_curiosity_flounder) (fears_abrasion_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_flounder)))
 (:action succumb__abrasion__curiosity__apple
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_apple) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_apple)))
 (:action succumb__abrasion__curiosity__hamburger
  :parameters ()
  :precondition (and (craves_curiosity_hamburger) (fears_abrasion_curiosity) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_hamburger)))
 (:action succumb__abrasion__curiosity__haroset
  :parameters ()
  :precondition (and (fears_abrasion_curiosity) (craves_curiosity_haroset) (not (of)))
  :effect (and (when (z_8_0) (v10_0)) (when (not (z_8_0)) (not (v10_0))) (when (z_8_1) (v10_1)) (when (not (z_8_1)) (not (v10_1))) (when (z_8_2) (v10_2)) (when (not (z_8_2)) (not (v10_2))) (when (z_8_3) (v10_3)) (when (not (z_8_3)) (not (v10_3))) (when (z_8_4) (v10_4)) (when (not (z_8_4)) (not (v10_4))) (when (and (not (v10_4)) (z_8_4)) (of)) (not (fears_abrasion_curiosity)) (craves_abrasion_haroset)))
 (:action succumb__abrasion__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_abrasion_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_abrasion_wurst) (not (fears_abrasion_intoxication))))
 (:action succumb__abrasion__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_abrasion_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_abrasion_hotdog) (not (fears_abrasion_intoxication))))
 (:action overcome__dread__entertainment__guava
  :parameters ()
  :precondition (and (craves_entertainment_guava) (not (v0_4)) (craves_dread_guava) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_dread_entertainment) (not (craves_dread_guava))))
 (:action succumb__anxiety__intoxication__guava
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_guava) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_guava)))
 (:action overcome__dread__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (not (v0_4)) (craves_dread_apple) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_dread_apple)) (fears_dread_entertainment)))
 (:action succumb__anxiety__intoxication__flounder
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_flounder) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_anxiety_flounder) (not (fears_anxiety_intoxication))))
 (:action succumb__anxiety__intoxication__apple
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_apple) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_apple)))
 (:action succumb__anxiety__intoxication__hamburger
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_hamburger) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_anxiety_hamburger) (not (fears_anxiety_intoxication))))
 (:action succumb__anxiety__intoxication__haroset
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_haroset) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_haroset)))
 (:action succumb__anxiety__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_hotdog)))
 (:action succumb__anxiety__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_anxiety_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_wurst)))
 (:action overcome__abrasion__excitement__wurst
  :parameters ()
  :precondition (and (craves_abrasion_wurst) (craves_excitement_wurst) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_abrasion_wurst)) (fears_abrasion_excitement)))
 (:action overcome__dread__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_dread_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_dread_curiosity) (not (craves_dread_guava))))
 (:action overcome__dread__curiosity__apple
  :parameters ()
  :precondition (and (not (v10_4)) (craves_curiosity_apple) (craves_dread_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_dread_apple)) (fears_dread_curiosity)))
 (:action overcome__dread__intoxication__guava
  :parameters ()
  :precondition (and (craves_dread_guava) (craves_intoxication_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_dread_intoxication) (not (craves_dread_guava))))
 (:action overcome__dread__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (not (v1_4)) (craves_dread_apple) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_dread_apple)) (fears_dread_intoxication)))
 (:action overcome__abrasion__entertainment__wurst
  :parameters ()
  :precondition (and (craves_abrasion_wurst) (not (v0_4)) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_abrasion_wurst)) (fears_abrasion_entertainment)))
 (:action overcome__loneliness__excitement__wurst
  :parameters ()
  :precondition (and (craves_loneliness_wurst) (craves_excitement_wurst) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_loneliness_excitement) (not (craves_loneliness_wurst))))
 (:action overcome__abrasion__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (craves_abrasion_wurst) (not (v10_4)) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_abrasion_curiosity) (not (craves_abrasion_wurst))))
 (:action succumb__anxiety__excitement__flounder
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_flounder) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anxiety_flounder) (not (fears_anxiety_excitement))))
 (:action succumb__anxiety__excitement__apple
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anxiety_apple) (not (fears_anxiety_excitement))))
 (:action succumb__anxiety__excitement__hamburger
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anxiety_hamburger) (not (fears_anxiety_excitement))))
 (:action succumb__anxiety__excitement__haroset
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anxiety_haroset) (not (fears_anxiety_excitement))))
 (:action succumb__anxiety__excitement__hotdog
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_hotdog) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anxiety_hotdog) (not (fears_anxiety_excitement))))
 (:action succumb__anxiety__excitement__wurst
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_wurst) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (not (fears_anxiety_excitement)) (craves_anxiety_wurst)))
 (:action succumb__anxiety__excitement__guava
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_anxiety_guava) (not (fears_anxiety_excitement))))
 (:action overcome__abrasion__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_abrasion_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_abrasion_wurst)) (fears_abrasion_intoxication)))
 (:action overcome__loneliness__entertainment__wurst
  :parameters ()
  :precondition (and (craves_loneliness_wurst) (not (v0_4)) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_loneliness_entertainment) (not (craves_loneliness_wurst))))
 (:action overcome__hangover__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (not (v4_4)) (craves_hangover_wurst) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_hangover_wurst)) (fears_hangover_excitement)))
 (:action overcome__loneliness__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (craves_loneliness_wurst) (not (v10_4)) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_loneliness_curiosity) (not (craves_loneliness_wurst))))
 (:action overcome__loneliness__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_loneliness_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_loneliness_wurst)) (fears_loneliness_intoxication)))
 (:action overcome__hangover__entertainment__wurst
  :parameters ()
  :precondition (and (not (v0_4)) (craves_hangover_wurst) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_hangover_wurst)) (fears_hangover_entertainment)))
 (:action overcome__hangover__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v10_4)) (craves_hangover_wurst) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_hangover_wurst)) (fears_hangover_curiosity)))
 (:action overcome__hangover__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_hangover_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_hangover_wurst)) (fears_hangover_intoxication)))
 (:action overcome__jealousy__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (craves_jealousy_wurst) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_jealousy_wurst)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__entertainment__wurst
  :parameters ()
  :precondition (and (craves_jealousy_wurst) (not (v0_4)) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_jealousy_wurst)) (fears_jealousy_entertainment)))
 (:action overcome__grief__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (not (v4_4)) (craves_grief_wurst) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_grief_wurst)) (fears_grief_excitement)))
 (:action overcome__jealousy__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v10_4)) (craves_jealousy_wurst) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_jealousy_wurst)) (fears_jealousy_curiosity)))
 (:action overcome__jealousy__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_jealousy_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_jealousy_wurst)) (fears_jealousy_intoxication)))
 (:action overcome__grief__entertainment__wurst
  :parameters ()
  :precondition (and (not (v0_4)) (craves_entertainment_wurst) (craves_grief_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_grief_wurst)) (fears_grief_entertainment)))
 (:action overcome__grief__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v10_4)) (craves_grief_wurst) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_grief_wurst)) (fears_grief_curiosity)))
 (:action overcome__grief__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (not (v1_4)) (craves_grief_wurst) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_grief_wurst)) (fears_grief_intoxication)))
 (:action overcome__sciatica__curiosity__guava
  :parameters ()
  :precondition (and (not (v10_4)) (craves_sciatica_guava) (craves_curiosity_guava) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_guava))))
 (:action overcome__sciatica__curiosity__apple
  :parameters ()
  :precondition (and (not (v10_4)) (craves_curiosity_apple) (craves_sciatica_apple) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_sciatica_curiosity) (not (craves_sciatica_apple))))
 (:action succumb__laceration__excitement__apple
  :parameters ()
  :precondition (and (fears_laceration_excitement) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_laceration_apple) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__haroset
  :parameters ()
  :precondition (and (fears_laceration_excitement) (craves_excitement_haroset) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_laceration_haroset) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__flounder
  :parameters ()
  :precondition (and (fears_laceration_excitement) (craves_excitement_flounder) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_laceration_flounder) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__wurst
  :parameters ()
  :precondition (and (fears_laceration_excitement) (craves_excitement_wurst) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_laceration_wurst) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__hamburger
  :parameters ()
  :precondition (and (fears_laceration_excitement) (craves_excitement_hamburger) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_laceration_hamburger) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__guava
  :parameters ()
  :precondition (and (fears_laceration_excitement) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_laceration_guava) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__hotdog
  :parameters ()
  :precondition (and (fears_laceration_excitement) (craves_excitement_hotdog) (not (of)))
  :effect (and (when (z_20_0) (v4_0)) (when (not (z_20_0)) (not (v4_0))) (when (z_20_1) (v4_1)) (when (not (z_20_1)) (not (v4_1))) (when (z_20_2) (v4_2)) (when (not (z_20_2)) (not (v4_2))) (when (z_20_3) (v4_3)) (when (not (z_20_3)) (not (v4_3))) (when (z_20_4) (v4_4)) (when (not (z_20_4)) (not (v4_4))) (when (and (not (v4_4)) (z_20_4)) (of)) (craves_laceration_hotdog) (not (fears_laceration_excitement))))
 (:action overcome__sciatica__intoxication__guava
  :parameters ()
  :precondition (and (craves_sciatica_guava) (craves_intoxication_guava) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_sciatica_guava)) (fears_sciatica_intoxication)))
 (:action overcome__sciatica__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (not (v1_4)) (craves_sciatica_apple) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_sciatica_apple)) (fears_sciatica_intoxication)))
 (:action succumb__laceration__intoxication__guava
  :parameters ()
  :precondition (and (craves_intoxication_guava) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_laceration_guava) (not (fears_laceration_intoxication))))
 (:action succumb__laceration__intoxication__hotdog
  :parameters ()
  :precondition (and (fears_laceration_intoxication) (craves_intoxication_hotdog) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_laceration_hotdog) (not (fears_laceration_intoxication))))
 (:action succumb__laceration__intoxication__apple
  :parameters ()
  :precondition (and (craves_intoxication_apple) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_laceration_apple) (not (fears_laceration_intoxication))))
 (:action succumb__laceration__intoxication__haroset
  :parameters ()
  :precondition (and (craves_intoxication_haroset) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_laceration_haroset) (not (fears_laceration_intoxication))))
 (:action succumb__laceration__intoxication__flounder
  :parameters ()
  :precondition (and (craves_intoxication_flounder) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_laceration_flounder) (not (fears_laceration_intoxication))))
 (:action succumb__laceration__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (not (fears_laceration_intoxication)) (craves_laceration_wurst)))
 (:action succumb__laceration__intoxication__hamburger
  :parameters ()
  :precondition (and (craves_intoxication_hamburger) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_17_0) (v1_0)) (when (not (z_17_0)) (not (v1_0))) (when (z_17_1) (v1_1)) (when (not (z_17_1)) (not (v1_1))) (when (z_17_2) (v1_2)) (when (not (z_17_2)) (not (v1_2))) (when (z_17_3) (v1_3)) (when (not (z_17_3)) (not (v1_3))) (when (z_17_4) (v1_4)) (when (not (z_17_4)) (not (v1_4))) (when (and (not (v1_4)) (z_17_4)) (of)) (craves_laceration_hamburger) (not (fears_laceration_intoxication))))
 (:action overcome__sciatica__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_sciatica_guava) (craves_excitement_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_guava))))
 (:action overcome__sciatica__excitement__apple
  :parameters ()
  :precondition (and (not (v4_4)) (craves_sciatica_apple) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_sciatica_apple)) (fears_sciatica_excitement)))
 (:action overcome__prostatitis__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v10_4)) (craves_prostatitis_wurst) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (not (craves_prostatitis_wurst)) (fears_prostatitis_curiosity)))
 (:action overcome__prostatitis__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_prostatitis_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_prostatitis_wurst)) (fears_prostatitis_intoxication)))
 (:action overcome__sciatica__entertainment__guava
  :parameters ()
  :precondition (and (craves_sciatica_guava) (craves_entertainment_guava) (not (v0_4)) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_sciatica_guava)) (fears_sciatica_entertainment)))
 (:action overcome__sciatica__entertainment__apple
  :parameters ()
  :precondition (and (craves_entertainment_apple) (not (v0_4)) (craves_sciatica_apple) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_sciatica_apple)) (fears_sciatica_entertainment)))
 (:action overcome__prostatitis__excitement__wurst
  :parameters ()
  :precondition (and (craves_prostatitis_wurst) (craves_excitement_wurst) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_prostatitis_wurst)) (fears_prostatitis_excitement)))
 (:action overcome__anger__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (not (v10_4)) (craves_anger_wurst) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_anger_curiosity) (not (craves_anger_wurst))))
 (:action overcome__anger__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_anger_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_anger_intoxication) (not (craves_anger_wurst))))
 (:action overcome__prostatitis__entertainment__wurst
  :parameters ()
  :precondition (and (craves_prostatitis_wurst) (not (v0_4)) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_prostatitis_wurst)) (fears_prostatitis_entertainment)))
 (:action overcome__anger__excitement__wurst
  :parameters ()
  :precondition (and (craves_excitement_wurst) (not (v4_4)) (craves_anger_wurst) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_anger_wurst)) (fears_anger_excitement)))
 (:action overcome__anger__entertainment__wurst
  :parameters ()
  :precondition (and (not (v0_4)) (craves_anger_wurst) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_anger_entertainment) (not (craves_anger_wurst))))
 (:action overcome__boils__curiosity__wurst
  :parameters ()
  :precondition (and (craves_curiosity_wurst) (craves_boils_wurst) (not (v10_4)) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_boils_curiosity) (not (craves_boils_wurst))))
 (:action overcome__boils__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_boils_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (fears_boils_intoxication) (not (craves_boils_wurst))))
 (:action overcome__boils__excitement__wurst
  :parameters ()
  :precondition (and (craves_boils_wurst) (craves_excitement_wurst) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_boils_excitement) (not (craves_boils_wurst))))
 (:action overcome__angina__curiosity__wurst
  :parameters ()
  :precondition (and (craves_angina_wurst) (craves_curiosity_wurst) (not (v10_4)) (not (of)))
  :effect (and (when (z_15_0) (v10_0)) (when (not (z_15_0)) (not (v10_0))) (when (z_15_1) (v10_1)) (when (not (z_15_1)) (not (v10_1))) (when (z_15_2) (v10_2)) (when (not (z_15_2)) (not (v10_2))) (when (z_15_3) (v10_3)) (when (not (z_15_3)) (not (v10_3))) (when (z_15_4) (v10_4)) (when (not (z_15_4)) (not (v10_4))) (when (and (v10_4) (not (z_15_4))) (of)) (fears_angina_curiosity) (not (craves_angina_wurst))))
 (:action overcome__angina__intoxication__wurst
  :parameters ()
  :precondition (and (craves_intoxication_wurst) (craves_angina_wurst) (not (v1_4)) (not (of)))
  :effect (and (when (z_4_0) (v1_0)) (when (not (z_4_0)) (not (v1_0))) (when (z_4_1) (v1_1)) (when (not (z_4_1)) (not (v1_1))) (when (z_4_2) (v1_2)) (when (not (z_4_2)) (not (v1_2))) (when (z_4_3) (v1_3)) (when (not (z_4_3)) (not (v1_3))) (when (z_4_4) (v1_4)) (when (not (z_4_4)) (not (v1_4))) (when (and (v1_4) (not (z_4_4))) (of)) (not (craves_angina_wurst)) (fears_angina_intoxication)))
 (:action overcome__boils__entertainment__wurst
  :parameters ()
  :precondition (and (craves_boils_wurst) (not (v0_4)) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (not (craves_boils_wurst)) (fears_boils_entertainment)))
 (:action overcome__angina__excitement__wurst
  :parameters ()
  :precondition (and (craves_angina_wurst) (craves_excitement_wurst) (not (v4_4)) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_angina_excitement) (not (craves_angina_wurst))))
 (:action overcome__angina__entertainment__wurst
  :parameters ()
  :precondition (and (craves_angina_wurst) (not (v0_4)) (craves_entertainment_wurst) (not (of)))
  :effect (and (when (z_2_0) (v0_0)) (when (not (z_2_0)) (not (v0_0))) (when (z_2_1) (v0_1)) (when (not (z_2_1)) (not (v0_1))) (when (z_2_2) (v0_2)) (when (not (z_2_2)) (not (v0_2))) (when (z_2_3) (v0_3)) (when (not (z_2_3)) (not (v0_3))) (when (z_2_4) (v0_4)) (when (not (z_2_4)) (not (v0_4))) (when (and (v0_4) (not (z_2_4))) (of)) (fears_angina_entertainment) (not (craves_angina_wurst))))
 (:action overcome__dread__excitement__apple
  :parameters ()
  :precondition (and (not (v4_4)) (craves_dread_apple) (craves_excitement_apple) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (not (craves_dread_apple)) (fears_dread_excitement)))
 (:action overcome__dread__excitement__guava
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_guava) (craves_dread_guava) (not (of)))
  :effect (and (when (z_9_0) (v4_0)) (when (not (z_9_0)) (not (v4_0))) (when (z_9_1) (v4_1)) (when (not (z_9_1)) (not (v4_1))) (when (z_9_2) (v4_2)) (when (not (z_9_2)) (not (v4_2))) (when (z_9_3) (v4_3)) (when (not (z_9_3)) (not (v4_3))) (when (z_9_4) (v4_4)) (when (not (z_9_4)) (not (v4_4))) (when (and (v4_4) (not (z_9_4))) (of)) (fears_dread_excitement) (not (craves_dread_guava))))
)
