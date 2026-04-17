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
             (c_22_-1)
             (c_22_0)
             (c_22_1)
             (c_22_2)
             (c_22_3)
             (c_22_4)
             (z_22_0)
             (z_22_1)
             (z_22_2)
             (z_22_3)
             (z_22_4)
             (c_23_-1)
             (c_23_0)
             (c_23_1)
             (c_23_2)
             (c_23_3)
             (c_23_4)
             (z_23_0)
             (z_23_1)
             (z_23_2)
             (z_23_3)
             (z_23_4)
             (false)
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
             (locale_turkey_0)
             (locale_turkey_1)
             (locale_turkey_2)
             (locale_turkey_3)
             (locale_turkey_4)
             (v6_0)
             (v6_1)
             (v6_2)
             (v6_3)
             (v6_4)
             (harmony_satisfaction_0)
             (harmony_satisfaction_1)
             (harmony_satisfaction_2)
             (harmony_satisfaction_3)
             (harmony_satisfaction_4)
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
             (harmony_excitement_0)
             (harmony_excitement_1)
             (harmony_excitement_2)
             (harmony_excitement_3)
             (harmony_excitement_4)
             (v9_0)
             (v9_1)
             (v9_2)
             (v9_3)
             (v9_4)
             (locale_broccoli_0)
             (locale_broccoli_1)
             (locale_broccoli_2)
             (locale_broccoli_3)
             (locale_broccoli_4)
             (locale_shrimp_0)
             (locale_shrimp_1)
             (locale_shrimp_2)
             (locale_shrimp_3)
             (locale_shrimp_4)
             (locale_sweetroll_0)
             (locale_sweetroll_1)
             (locale_sweetroll_2)
             (locale_sweetroll_3)
             (locale_sweetroll_4)
             (locale_scallop_0)
             (locale_scallop_1)
             (locale_scallop_2)
             (locale_scallop_3)
             (locale_scallop_4)
             (locale_cherry_0)
             (locale_cherry_1)
             (locale_cherry_2)
             (locale_cherry_3)
             (locale_cherry_4)
             (locale_chocolate_0)
             (locale_chocolate_1)
             (locale_chocolate_2)
             (locale_chocolate_3)
             (locale_chocolate_4)
             (locale_tuna_0)
             (locale_tuna_1)
             (locale_tuna_2)
             (locale_tuna_3)
             (locale_tuna_4)
             (v10_0)
             (v10_1)
             (v10_2)
             (v10_3)
             (v10_4)
             (harmony_lubricity_0)
             (harmony_lubricity_1)
             (harmony_lubricity_2)
             (harmony_lubricity_3)
             (harmony_lubricity_4)
             (v11_0)
             (v11_1)
             (v11_2)
             (v11_3)
             (v11_4)
             (harmony_intoxication_0)
             (harmony_intoxication_1)
             (harmony_intoxication_2)
             (harmony_intoxication_3)
             (harmony_intoxication_4)
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
             (v2_0)
             (v2_1)
             (v2_2)
             (v2_3)
             (v2_4)
             (v3_0)
             (v3_1)
             (v3_2)
             (v3_3)
             (v3_4)
             (craves_satisfaction_broccoli)
             (craves_jealousy_shrimp)
             (craves_grief_broccoli)
             (craves_intoxication_tuna)
             (craves_hangover_scallop)
             (craves_excitement_turkey)
             (craves_loneliness_scallop)
             (craves_depression_turkey)
             (craves_lubricity_sweetroll)
             (craves_dread_scallop)
             (craves_grief2_scallop)
             (craves_angina_shrimp)
             (craves_laceration_shrimp)
             (craves_anxiety_broccoli)
             (craves_prostatitis_turkey)
             (craves_boils_broccoli)
             (craves_abrasion_turkey)
             (craves_anger_cherry)
             (craves_sciatica_broccoli)
             (craves_loneliness_shrimp)
             (craves_grief_shrimp)
             (craves_satisfaction_scallop)
             (craves_satisfaction_cherry)
             (craves_excitement_scallop)
             (craves_excitement_cherry)
             (craves_intoxication_scallop)
             (craves_intoxication_cherry)
             (craves_lubricity_scallop)
             (craves_lubricity_cherry)
             (craves_satisfaction_tuna)
             (craves_excitement_tuna)
             (craves_excitement_broccoli)
             (craves_intoxication_broccoli)
             (craves_lubricity_tuna)
             (craves_lubricity_broccoli)
             (craves_satisfaction_sweetroll)
             (craves_excitement_sweetroll)
             (craves_intoxication_sweetroll)
             (craves_satisfaction_shrimp)
             (craves_excitement_shrimp)
             (craves_intoxication_shrimp)
             (craves_lubricity_shrimp)
             (craves_satisfaction_chocolate)
             (craves_excitement_chocolate)
             (craves_intoxication_chocolate)
             (craves_lubricity_chocolate)
             (craves_satisfaction_turkey)
             (craves_intoxication_turkey)
             (craves_lubricity_turkey)
             (fears_depression_lubricity)
             (craves_depression_broccoli)
             (fears_boils_excitement)
             (craves_depression_chocolate)
             (craves_boils_chocolate)
             (craves_boils_turkey)
             (craves_depression_tuna)
             (craves_boils_tuna)
             (craves_depression_sweetroll)
             (craves_boils_sweetroll)
             (craves_depression_shrimp)
             (craves_boils_shrimp)
             (craves_depression_cherry)
             (craves_boils_cherry)
             (craves_depression_scallop)
             (craves_boils_scallop)
             (fears_boils_intoxication)
             (fears_grief_satisfaction)
             (craves_grief_chocolate)
             (craves_grief_turkey)
             (craves_grief_tuna)
             (craves_grief_sweetroll)
             (craves_grief_cherry)
             (craves_grief_scallop)
             (fears_boils_lubricity)
             (fears_grief_excitement)
             (fears_grief_intoxication)
             (fears_anxiety_satisfaction)
             (craves_anxiety_chocolate)
             (craves_anxiety_turkey)
             (craves_anxiety_tuna)
             (craves_anxiety_sweetroll)
             (craves_anxiety_shrimp)
             (craves_anxiety_cherry)
             (craves_anxiety_scallop)
             (fears_grief_lubricity)
             (fears_anxiety_excitement)
             (fears_anxiety_intoxication)
             (fears_sciatica_satisfaction)
             (craves_sciatica_chocolate)
             (craves_sciatica_turkey)
             (craves_sciatica_tuna)
             (craves_sciatica_sweetroll)
             (craves_sciatica_shrimp)
             (craves_sciatica_cherry)
             (craves_sciatica_scallop)
             (fears_anxiety_lubricity)
             (fears_sciatica_excitement)
             (fears_sciatica_intoxication)
             (fears_sciatica_lubricity)
             (fears_hangover_satisfaction)
             (craves_hangover_broccoli)
             (craves_hangover_turkey)
             (craves_hangover_chocolate)
             (craves_hangover_sweetroll)
             (craves_hangover_tuna)
             (craves_hangover_cherry)
             (craves_hangover_shrimp)
             (fears_hangover_excitement)
             (fears_hangover_intoxication)
             (fears_loneliness_satisfaction)
             (craves_loneliness_chocolate)
             (craves_loneliness_broccoli)
             (craves_loneliness_tuna)
             (craves_loneliness_turkey)
             (craves_loneliness_sweetroll)
             (craves_loneliness_cherry)
             (fears_loneliness_excitement)
             (fears_hangover_lubricity)
             (fears_dread_satisfaction)
             (craves_dread_broccoli)
             (fears_loneliness_intoxication)
             (craves_dread_turkey)
             (craves_dread_chocolate)
             (craves_dread_sweetroll)
             (craves_dread_tuna)
             (craves_dread_cherry)
             (craves_dread_shrimp)
             (fears_loneliness_lubricity)
             (fears_dread_excitement)
             (fears_dread_intoxication)
             (fears_grief2_satisfaction)
             (craves_grief2_chocolate)
             (craves_grief2_broccoli)
             (craves_grief2_tuna)
             (craves_grief2_turkey)
             (craves_grief2_shrimp)
             (craves_grief2_sweetroll)
             (craves_grief2_cherry)
             (fears_grief2_excitement)
             (fears_dread_lubricity)
             (fears_grief2_intoxication)
             (fears_anger_satisfaction)
             (craves_anger_broccoli)
             (craves_anger_chocolate)
             (craves_anger_turkey)
             (craves_anger_tuna)
             (craves_anger_sweetroll)
             (craves_anger_shrimp)
             (craves_anger_scallop)
             (fears_grief2_lubricity)
             (fears_anger_excitement)
             (fears_anger_intoxication)
             (fears_laceration_satisfaction)
             (craves_laceration_broccoli)
             (craves_laceration_chocolate)
             (craves_laceration_turkey)
             (craves_laceration_tuna)
             (craves_laceration_sweetroll)
             (craves_laceration_cherry)
             (craves_laceration_scallop)
             (fears_anger_lubricity)
             (fears_laceration_excitement)
             (fears_laceration_intoxication)
             (fears_jealousy_satisfaction)
             (craves_jealousy_broccoli)
             (craves_jealousy_chocolate)
             (craves_jealousy_turkey)
             (craves_jealousy_tuna)
             (craves_jealousy_sweetroll)
             (craves_jealousy_cherry)
             (craves_jealousy_scallop)
             (fears_laceration_lubricity)
             (fears_jealousy_excitement)
             (fears_jealousy_intoxication)
             (fears_angina_satisfaction)
             (craves_angina_broccoli)
             (craves_angina_chocolate)
             (craves_angina_turkey)
             (craves_angina_tuna)
             (craves_angina_sweetroll)
             (craves_angina_cherry)
             (craves_angina_scallop)
             (fears_jealousy_lubricity)
             (fears_angina_excitement)
             (fears_angina_intoxication)
             (fears_prostatitis_satisfaction)
             (craves_prostatitis_broccoli)
             (craves_prostatitis_chocolate)
             (craves_prostatitis_tuna)
             (craves_prostatitis_sweetroll)
             (craves_prostatitis_shrimp)
             (craves_prostatitis_cherry)
             (craves_prostatitis_scallop)
             (fears_angina_lubricity)
             (fears_prostatitis_excitement)
             (fears_prostatitis_intoxication)
             (fears_abrasion_satisfaction)
             (craves_abrasion_broccoli)
             (craves_abrasion_chocolate)
             (craves_abrasion_tuna)
             (craves_abrasion_sweetroll)
             (craves_abrasion_shrimp)
             (craves_abrasion_cherry)
             (craves_abrasion_scallop)
             (fears_prostatitis_lubricity)
             (fears_abrasion_excitement)
             (fears_abrasion_intoxication)
             (fears_depression_satisfaction)
             (fears_abrasion_lubricity)
             (fears_depression_excitement)
             (fears_depression_intoxication)
             (fears_boils_satisfaction)
             (of)
 )
 (:derived (c_0_-1) (false))
(:derived (c_0_0) (or (v1_0) (and (c_0_-1) (not (v1_0)))))
(:derived (c_0_1) (or (v1_1) (and (c_0_0) (not (v1_1)))))
(:derived (c_0_2) (or (v1_2) (and (c_0_1) (not (v1_2)))))
(:derived (c_0_3) (or (v1_3) (and (c_0_2) (not (v1_3)))))
(:derived (c_0_4) (or (v1_4) (and (c_0_3) (not (v1_4)))))
(:derived (z_0_0) (or (and (not (v1_0)) (not (c_0_-1))) (and (c_0_-1) (v1_0))))
(:derived (z_0_1) (or (and (not (v1_1)) (not (c_0_0))) (and (c_0_0) (v1_1))))
(:derived (z_0_2) (or (and (not (v1_2)) (not (c_0_1))) (and (c_0_1) (v1_2))))
(:derived (z_0_3) (or (and (not (v1_3)) (not (c_0_2))) (and (c_0_2) (v1_3))))
(:derived (z_0_4) (or (and (not (v1_4)) (not (c_0_3))) (and (c_0_3) (v1_4))))
(:derived (c_1_-1) (false))
(:derived (c_1_0) (or (v2_0) (and (c_1_-1) (not (v2_0)))))
(:derived (c_1_1) (or (v2_1) (and (c_1_0) (not (v2_1)))))
(:derived (c_1_2) (or (v2_2) (and (c_1_1) (not (v2_2)))))
(:derived (c_1_3) (or (v2_3) (and (c_1_2) (not (v2_3)))))
(:derived (c_1_4) (or (v2_4) (and (c_1_3) (not (v2_4)))))
(:derived (z_1_0) (or (and (not (v2_0)) (not (c_1_-1))) (and (c_1_-1) (v2_0))))
(:derived (z_1_1) (or (and (not (v2_1)) (not (c_1_0))) (and (c_1_0) (v2_1))))
(:derived (z_1_2) (or (and (not (v2_2)) (not (c_1_1))) (and (c_1_1) (v2_2))))
(:derived (z_1_3) (or (and (not (v2_3)) (not (c_1_2))) (and (c_1_2) (v2_3))))
(:derived (z_1_4) (or (and (not (v2_4)) (not (c_1_3))) (and (c_1_3) (v2_4))))
(:derived (c_2_-1) (false))
(:derived (c_2_0) (or (v3_0) (and (c_2_-1) (not (v3_0)))))
(:derived (c_2_1) (or (v3_1) (and (c_2_0) (not (v3_1)))))
(:derived (c_2_2) (or (v3_2) (and (c_2_1) (not (v3_2)))))
(:derived (c_2_3) (or (v3_3) (and (c_2_2) (not (v3_3)))))
(:derived (c_2_4) (or (v3_4) (and (c_2_3) (not (v3_4)))))
(:derived (z_2_0) (or (and (not (v3_0)) (not (c_2_-1))) (and (c_2_-1) (v3_0))))
(:derived (z_2_1) (or (and (not (v3_1)) (not (c_2_0))) (and (c_2_0) (v3_1))))
(:derived (z_2_2) (or (and (not (v3_2)) (not (c_2_1))) (and (c_2_1) (v3_2))))
(:derived (z_2_3) (or (and (not (v3_3)) (not (c_2_2))) (and (c_2_2) (v3_3))))
(:derived (z_2_4) (or (and (not (v3_4)) (not (c_2_3))) (and (c_2_3) (v3_4))))
(:derived (c_3_-1) (false))
(:derived (c_3_0) (or (v4_0) (and (c_3_-1) (not (v4_0)))))
(:derived (c_3_1) (or (v4_1) (and (c_3_0) (not (v4_1)))))
(:derived (c_3_2) (or (v4_2) (and (c_3_1) (not (v4_2)))))
(:derived (c_3_3) (or (v4_3) (and (c_3_2) (not (v4_3)))))
(:derived (c_3_4) (or (v4_4) (and (c_3_3) (not (v4_4)))))
(:derived (z_3_0) (or (and (not (v4_0)) (not (c_3_-1))) (and (c_3_-1) (v4_0))))
(:derived (z_3_1) (or (and (not (v4_1)) (not (c_3_0))) (and (c_3_0) (v4_1))))
(:derived (z_3_2) (or (and (not (v4_2)) (not (c_3_1))) (and (c_3_1) (v4_2))))
(:derived (z_3_3) (or (and (not (v4_3)) (not (c_3_2))) (and (c_3_2) (v4_3))))
(:derived (z_3_4) (or (and (not (v4_4)) (not (c_3_3))) (and (c_3_3) (v4_4))))
(:derived (c_4_-1) (false))
(:derived (c_4_0) (or (v0_0) (and (c_4_-1) (not (v0_0)))))
(:derived (c_4_1) (and (c_4_0) (v0_1)))
(:derived (c_4_2) (and (c_4_1) (v0_2)))
(:derived (c_4_3) (and (c_4_2) (v0_3)))
(:derived (c_4_4) (and (c_4_3) (v0_4)))
(:derived (z_4_0) (or (and (not (v0_0)) (not (c_4_-1))) (and (c_4_-1) (v0_0))))
(:derived (z_4_1) (or (and (v0_1) (not (c_4_0))) (and (c_4_0) (not (v0_1)))))
(:derived (z_4_2) (or (and (v0_2) (not (c_4_1))) (and (c_4_1) (not (v0_2)))))
(:derived (z_4_3) (or (and (v0_3) (not (c_4_2))) (and (c_4_2) (not (v0_3)))))
(:derived (z_4_4) (or (and (v0_4) (not (c_4_3))) (and (c_4_3) (not (v0_4)))))
(:derived (c_5_-1) (false))
(:derived (c_5_0) (or (v5_0) (and (c_5_-1) (not (v5_0)))))
(:derived (c_5_1) (or (v5_1) (and (c_5_0) (not (v5_1)))))
(:derived (c_5_2) (or (v5_2) (and (c_5_1) (not (v5_2)))))
(:derived (c_5_3) (or (v5_3) (and (c_5_2) (not (v5_3)))))
(:derived (c_5_4) (or (v5_4) (and (c_5_3) (not (v5_4)))))
(:derived (z_5_0) (or (and (not (v5_0)) (not (c_5_-1))) (and (c_5_-1) (v5_0))))
(:derived (z_5_1) (or (and (not (v5_1)) (not (c_5_0))) (and (c_5_0) (v5_1))))
(:derived (z_5_2) (or (and (not (v5_2)) (not (c_5_1))) (and (c_5_1) (v5_2))))
(:derived (z_5_3) (or (and (not (v5_3)) (not (c_5_2))) (and (c_5_2) (v5_3))))
(:derived (z_5_4) (or (and (not (v5_4)) (not (c_5_3))) (and (c_5_3) (v5_4))))
(:derived (c_6_-1) (false))
(:derived (c_6_0) (or (v1_0) (and (c_6_-1) (not (v1_0)))))
(:derived (c_6_1) (and (c_6_0) (v1_1)))
(:derived (c_6_2) (and (c_6_1) (v1_2)))
(:derived (c_6_3) (and (c_6_2) (v1_3)))
(:derived (c_6_4) (and (c_6_3) (v1_4)))
(:derived (z_6_0) (or (and (not (v1_0)) (not (c_6_-1))) (and (c_6_-1) (v1_0))))
(:derived (z_6_1) (or (and (v1_1) (not (c_6_0))) (and (c_6_0) (not (v1_1)))))
(:derived (z_6_2) (or (and (v1_2) (not (c_6_1))) (and (c_6_1) (not (v1_2)))))
(:derived (z_6_3) (or (and (v1_3) (not (c_6_2))) (and (c_6_2) (not (v1_3)))))
(:derived (z_6_4) (or (and (v1_4) (not (c_6_3))) (and (c_6_3) (not (v1_4)))))
(:derived (c_7_-1) (false))
(:derived (c_7_0) (or (v6_0) (and (c_7_-1) (not (v6_0)))))
(:derived (c_7_1) (or (v6_1) (and (c_7_0) (not (v6_1)))))
(:derived (c_7_2) (or (v6_2) (and (c_7_1) (not (v6_2)))))
(:derived (c_7_3) (or (v6_3) (and (c_7_2) (not (v6_3)))))
(:derived (c_7_4) (or (v6_4) (and (c_7_3) (not (v6_4)))))
(:derived (z_7_0) (or (and (not (v6_0)) (not (c_7_-1))) (and (c_7_-1) (v6_0))))
(:derived (z_7_1) (or (and (not (v6_1)) (not (c_7_0))) (and (c_7_0) (v6_1))))
(:derived (z_7_2) (or (and (not (v6_2)) (not (c_7_1))) (and (c_7_1) (v6_2))))
(:derived (z_7_3) (or (and (not (v6_3)) (not (c_7_2))) (and (c_7_2) (v6_3))))
(:derived (z_7_4) (or (and (not (v6_4)) (not (c_7_3))) (and (c_7_3) (v6_4))))
(:derived (c_8_-1) (false))
(:derived (c_8_0) (or (v2_0) (and (c_8_-1) (not (v2_0)))))
(:derived (c_8_1) (and (c_8_0) (v2_1)))
(:derived (c_8_2) (and (c_8_1) (v2_2)))
(:derived (c_8_3) (and (c_8_2) (v2_3)))
(:derived (c_8_4) (and (c_8_3) (v2_4)))
(:derived (z_8_0) (or (and (not (v2_0)) (not (c_8_-1))) (and (c_8_-1) (v2_0))))
(:derived (z_8_1) (or (and (v2_1) (not (c_8_0))) (and (c_8_0) (not (v2_1)))))
(:derived (z_8_2) (or (and (v2_2) (not (c_8_1))) (and (c_8_1) (not (v2_2)))))
(:derived (z_8_3) (or (and (v2_3) (not (c_8_2))) (and (c_8_2) (not (v2_3)))))
(:derived (z_8_4) (or (and (v2_4) (not (c_8_3))) (and (c_8_3) (not (v2_4)))))
(:derived (c_9_-1) (false))
(:derived (c_9_0) (or (v7_0) (and (c_9_-1) (not (v7_0)))))
(:derived (c_9_1) (or (v7_1) (and (c_9_0) (not (v7_1)))))
(:derived (c_9_2) (or (v7_2) (and (c_9_1) (not (v7_2)))))
(:derived (c_9_3) (or (v7_3) (and (c_9_2) (not (v7_3)))))
(:derived (c_9_4) (or (v7_4) (and (c_9_3) (not (v7_4)))))
(:derived (z_9_0) (or (and (not (v7_0)) (not (c_9_-1))) (and (c_9_-1) (v7_0))))
(:derived (z_9_1) (or (and (not (v7_1)) (not (c_9_0))) (and (c_9_0) (v7_1))))
(:derived (z_9_2) (or (and (not (v7_2)) (not (c_9_1))) (and (c_9_1) (v7_2))))
(:derived (z_9_3) (or (and (not (v7_3)) (not (c_9_2))) (and (c_9_2) (v7_3))))
(:derived (z_9_4) (or (and (not (v7_4)) (not (c_9_3))) (and (c_9_3) (v7_4))))
(:derived (c_10_-1) (false))
(:derived (c_10_0) (or (v3_0) (and (c_10_-1) (not (v3_0)))))
(:derived (c_10_1) (and (c_10_0) (v3_1)))
(:derived (c_10_2) (and (c_10_1) (v3_2)))
(:derived (c_10_3) (and (c_10_2) (v3_3)))
(:derived (c_10_4) (and (c_10_3) (v3_4)))
(:derived (z_10_0) (or (and (not (v3_0)) (not (c_10_-1))) (and (c_10_-1) (v3_0))))
(:derived (z_10_1) (or (and (v3_1) (not (c_10_0))) (and (c_10_0) (not (v3_1)))))
(:derived (z_10_2) (or (and (v3_2) (not (c_10_1))) (and (c_10_1) (not (v3_2)))))
(:derived (z_10_3) (or (and (v3_3) (not (c_10_2))) (and (c_10_2) (not (v3_3)))))
(:derived (z_10_4) (or (and (v3_4) (not (c_10_3))) (and (c_10_3) (not (v3_4)))))
(:derived (c_11_-1) (false))
(:derived (c_11_0) (or (v8_0) (and (c_11_-1) (not (v8_0)))))
(:derived (c_11_1) (or (v8_1) (and (c_11_0) (not (v8_1)))))
(:derived (c_11_2) (or (v8_2) (and (c_11_1) (not (v8_2)))))
(:derived (c_11_3) (or (v8_3) (and (c_11_2) (not (v8_3)))))
(:derived (c_11_4) (or (v8_4) (and (c_11_3) (not (v8_4)))))
(:derived (z_11_0) (or (and (not (v8_0)) (not (c_11_-1))) (and (c_11_-1) (v8_0))))
(:derived (z_11_1) (or (and (not (v8_1)) (not (c_11_0))) (and (c_11_0) (v8_1))))
(:derived (z_11_2) (or (and (not (v8_2)) (not (c_11_1))) (and (c_11_1) (v8_2))))
(:derived (z_11_3) (or (and (not (v8_3)) (not (c_11_2))) (and (c_11_2) (v8_3))))
(:derived (z_11_4) (or (and (not (v8_4)) (not (c_11_3))) (and (c_11_3) (v8_4))))
(:derived (c_12_-1) (false))
(:derived (c_12_0) (or (v4_0) (and (c_12_-1) (not (v4_0)))))
(:derived (c_12_1) (and (c_12_0) (v4_1)))
(:derived (c_12_2) (and (c_12_1) (v4_2)))
(:derived (c_12_3) (and (c_12_2) (v4_3)))
(:derived (c_12_4) (and (c_12_3) (v4_4)))
(:derived (z_12_0) (or (and (not (v4_0)) (not (c_12_-1))) (and (c_12_-1) (v4_0))))
(:derived (z_12_1) (or (and (v4_1) (not (c_12_0))) (and (c_12_0) (not (v4_1)))))
(:derived (z_12_2) (or (and (v4_2) (not (c_12_1))) (and (c_12_1) (not (v4_2)))))
(:derived (z_12_3) (or (and (v4_3) (not (c_12_2))) (and (c_12_2) (not (v4_3)))))
(:derived (z_12_4) (or (and (v4_4) (not (c_12_3))) (and (c_12_3) (not (v4_4)))))
(:derived (c_13_-1) (false))
(:derived (c_13_0) (or (v9_0) (and (c_13_-1) (not (v9_0)))))
(:derived (c_13_1) (or (v9_1) (and (c_13_0) (not (v9_1)))))
(:derived (c_13_2) (or (v9_2) (and (c_13_1) (not (v9_2)))))
(:derived (c_13_3) (or (v9_3) (and (c_13_2) (not (v9_3)))))
(:derived (c_13_4) (or (v9_4) (and (c_13_3) (not (v9_4)))))
(:derived (z_13_0) (or (and (not (v9_0)) (not (c_13_-1))) (and (c_13_-1) (v9_0))))
(:derived (z_13_1) (or (and (not (v9_1)) (not (c_13_0))) (and (c_13_0) (v9_1))))
(:derived (z_13_2) (or (and (not (v9_2)) (not (c_13_1))) (and (c_13_1) (v9_2))))
(:derived (z_13_3) (or (and (not (v9_3)) (not (c_13_2))) (and (c_13_2) (v9_3))))
(:derived (z_13_4) (or (and (not (v9_4)) (not (c_13_3))) (and (c_13_3) (v9_4))))
(:derived (c_14_-1) (false))
(:derived (c_14_0) (or (v5_0) (and (c_14_-1) (not (v5_0)))))
(:derived (c_14_1) (and (c_14_0) (v5_1)))
(:derived (c_14_2) (and (c_14_1) (v5_2)))
(:derived (c_14_3) (and (c_14_2) (v5_3)))
(:derived (c_14_4) (and (c_14_3) (v5_4)))
(:derived (z_14_0) (or (and (not (v5_0)) (not (c_14_-1))) (and (c_14_-1) (v5_0))))
(:derived (z_14_1) (or (and (v5_1) (not (c_14_0))) (and (c_14_0) (not (v5_1)))))
(:derived (z_14_2) (or (and (v5_2) (not (c_14_1))) (and (c_14_1) (not (v5_2)))))
(:derived (z_14_3) (or (and (v5_3) (not (c_14_2))) (and (c_14_2) (not (v5_3)))))
(:derived (z_14_4) (or (and (v5_4) (not (c_14_3))) (and (c_14_3) (not (v5_4)))))
(:derived (c_15_-1) (false))
(:derived (c_15_0) (or (v6_0) (and (c_15_-1) (not (v6_0)))))
(:derived (c_15_1) (and (c_15_0) (v6_1)))
(:derived (c_15_2) (and (c_15_1) (v6_2)))
(:derived (c_15_3) (and (c_15_2) (v6_3)))
(:derived (c_15_4) (and (c_15_3) (v6_4)))
(:derived (z_15_0) (or (and (not (v6_0)) (not (c_15_-1))) (and (c_15_-1) (v6_0))))
(:derived (z_15_1) (or (and (v6_1) (not (c_15_0))) (and (c_15_0) (not (v6_1)))))
(:derived (z_15_2) (or (and (v6_2) (not (c_15_1))) (and (c_15_1) (not (v6_2)))))
(:derived (z_15_3) (or (and (v6_3) (not (c_15_2))) (and (c_15_2) (not (v6_3)))))
(:derived (z_15_4) (or (and (v6_4) (not (c_15_3))) (and (c_15_3) (not (v6_4)))))
(:derived (c_16_-1) (false))
(:derived (c_16_0) (or (v7_0) (and (c_16_-1) (not (v7_0)))))
(:derived (c_16_1) (and (c_16_0) (v7_1)))
(:derived (c_16_2) (and (c_16_1) (v7_2)))
(:derived (c_16_3) (and (c_16_2) (v7_3)))
(:derived (c_16_4) (and (c_16_3) (v7_4)))
(:derived (z_16_0) (or (and (not (v7_0)) (not (c_16_-1))) (and (c_16_-1) (v7_0))))
(:derived (z_16_1) (or (and (v7_1) (not (c_16_0))) (and (c_16_0) (not (v7_1)))))
(:derived (z_16_2) (or (and (v7_2) (not (c_16_1))) (and (c_16_1) (not (v7_2)))))
(:derived (z_16_3) (or (and (v7_3) (not (c_16_2))) (and (c_16_2) (not (v7_3)))))
(:derived (z_16_4) (or (and (v7_4) (not (c_16_3))) (and (c_16_3) (not (v7_4)))))
(:derived (c_17_-1) (false))
(:derived (c_17_0) (or (v10_0) (and (c_17_-1) (not (v10_0)))))
(:derived (c_17_1) (or (v10_1) (and (c_17_0) (not (v10_1)))))
(:derived (c_17_2) (or (v10_2) (and (c_17_1) (not (v10_2)))))
(:derived (c_17_3) (or (v10_3) (and (c_17_2) (not (v10_3)))))
(:derived (c_17_4) (or (v10_4) (and (c_17_3) (not (v10_4)))))
(:derived (z_17_0) (or (and (not (v10_0)) (not (c_17_-1))) (and (c_17_-1) (v10_0))))
(:derived (z_17_1) (or (and (not (v10_1)) (not (c_17_0))) (and (c_17_0) (v10_1))))
(:derived (z_17_2) (or (and (not (v10_2)) (not (c_17_1))) (and (c_17_1) (v10_2))))
(:derived (z_17_3) (or (and (not (v10_3)) (not (c_17_2))) (and (c_17_2) (v10_3))))
(:derived (z_17_4) (or (and (not (v10_4)) (not (c_17_3))) (and (c_17_3) (v10_4))))
(:derived (c_18_-1) (false))
(:derived (c_18_0) (or (v8_0) (and (c_18_-1) (not (v8_0)))))
(:derived (c_18_1) (and (c_18_0) (v8_1)))
(:derived (c_18_2) (and (c_18_1) (v8_2)))
(:derived (c_18_3) (and (c_18_2) (v8_3)))
(:derived (c_18_4) (and (c_18_3) (v8_4)))
(:derived (z_18_0) (or (and (not (v8_0)) (not (c_18_-1))) (and (c_18_-1) (v8_0))))
(:derived (z_18_1) (or (and (v8_1) (not (c_18_0))) (and (c_18_0) (not (v8_1)))))
(:derived (z_18_2) (or (and (v8_2) (not (c_18_1))) (and (c_18_1) (not (v8_2)))))
(:derived (z_18_3) (or (and (v8_3) (not (c_18_2))) (and (c_18_2) (not (v8_3)))))
(:derived (z_18_4) (or (and (v8_4) (not (c_18_3))) (and (c_18_3) (not (v8_4)))))
(:derived (c_19_-1) (false))
(:derived (c_19_0) (or (v9_0) (and (c_19_-1) (not (v9_0)))))
(:derived (c_19_1) (and (c_19_0) (v9_1)))
(:derived (c_19_2) (and (c_19_1) (v9_2)))
(:derived (c_19_3) (and (c_19_2) (v9_3)))
(:derived (c_19_4) (and (c_19_3) (v9_4)))
(:derived (z_19_0) (or (and (not (v9_0)) (not (c_19_-1))) (and (c_19_-1) (v9_0))))
(:derived (z_19_1) (or (and (v9_1) (not (c_19_0))) (and (c_19_0) (not (v9_1)))))
(:derived (z_19_2) (or (and (v9_2) (not (c_19_1))) (and (c_19_1) (not (v9_2)))))
(:derived (z_19_3) (or (and (v9_3) (not (c_19_2))) (and (c_19_2) (not (v9_3)))))
(:derived (z_19_4) (or (and (v9_4) (not (c_19_3))) (and (c_19_3) (not (v9_4)))))
(:derived (c_20_-1) (false))
(:derived (c_20_0) (or (v10_0) (and (c_20_-1) (not (v10_0)))))
(:derived (c_20_1) (and (c_20_0) (v10_1)))
(:derived (c_20_2) (and (c_20_1) (v10_2)))
(:derived (c_20_3) (and (c_20_2) (v10_3)))
(:derived (c_20_4) (and (c_20_3) (v10_4)))
(:derived (z_20_0) (or (and (not (v10_0)) (not (c_20_-1))) (and (c_20_-1) (v10_0))))
(:derived (z_20_1) (or (and (v10_1) (not (c_20_0))) (and (c_20_0) (not (v10_1)))))
(:derived (z_20_2) (or (and (v10_2) (not (c_20_1))) (and (c_20_1) (not (v10_2)))))
(:derived (z_20_3) (or (and (v10_3) (not (c_20_2))) (and (c_20_2) (not (v10_3)))))
(:derived (z_20_4) (or (and (v10_4) (not (c_20_3))) (and (c_20_3) (not (v10_4)))))
(:derived (c_21_-1) (false))
(:derived (c_21_0) (or (v11_0) (and (c_21_-1) (not (v11_0)))))
(:derived (c_21_1) (or (v11_1) (and (c_21_0) (not (v11_1)))))
(:derived (c_21_2) (or (v11_2) (and (c_21_1) (not (v11_2)))))
(:derived (c_21_3) (or (v11_3) (and (c_21_2) (not (v11_3)))))
(:derived (c_21_4) (or (v11_4) (and (c_21_3) (not (v11_4)))))
(:derived (z_21_0) (or (and (not (v11_0)) (not (c_21_-1))) (and (c_21_-1) (v11_0))))
(:derived (z_21_1) (or (and (not (v11_1)) (not (c_21_0))) (and (c_21_0) (v11_1))))
(:derived (z_21_2) (or (and (not (v11_2)) (not (c_21_1))) (and (c_21_1) (v11_2))))
(:derived (z_21_3) (or (and (not (v11_3)) (not (c_21_2))) (and (c_21_2) (v11_3))))
(:derived (z_21_4) (or (and (not (v11_4)) (not (c_21_3))) (and (c_21_3) (v11_4))))
(:derived (c_22_-1) (false))
(:derived (c_22_0) (or (v11_0) (and (c_22_-1) (not (v11_0)))))
(:derived (c_22_1) (and (c_22_0) (v11_1)))
(:derived (c_22_2) (and (c_22_1) (v11_2)))
(:derived (c_22_3) (and (c_22_2) (v11_3)))
(:derived (c_22_4) (and (c_22_3) (v11_4)))
(:derived (z_22_0) (or (and (not (v11_0)) (not (c_22_-1))) (and (c_22_-1) (v11_0))))
(:derived (z_22_1) (or (and (v11_1) (not (c_22_0))) (and (c_22_0) (not (v11_1)))))
(:derived (z_22_2) (or (and (v11_2) (not (c_22_1))) (and (c_22_1) (not (v11_2)))))
(:derived (z_22_3) (or (and (v11_3) (not (c_22_2))) (and (c_22_2) (not (v11_3)))))
(:derived (z_22_4) (or (and (v11_4) (not (c_22_3))) (and (c_22_3) (not (v11_4)))))
(:derived (c_23_-1) (false))
(:derived (c_23_0) (or (v0_0) (and (c_23_-1) (not (v0_0)))))
(:derived (c_23_1) (or (v0_1) (and (c_23_0) (not (v0_1)))))
(:derived (c_23_2) (or (v0_2) (and (c_23_1) (not (v0_2)))))
(:derived (c_23_3) (or (v0_3) (and (c_23_2) (not (v0_3)))))
(:derived (c_23_4) (or (v0_4) (and (c_23_3) (not (v0_4)))))
(:derived (z_23_0) (or (and (not (v0_0)) (not (c_23_-1))) (and (c_23_-1) (v0_0))))
(:derived (z_23_1) (or (and (not (v0_1)) (not (c_23_0))) (and (c_23_0) (v0_1))))
(:derived (z_23_2) (or (and (not (v0_2)) (not (c_23_1))) (and (c_23_1) (v0_2))))
(:derived (z_23_3) (or (and (not (v0_3)) (not (c_23_2))) (and (c_23_2) (v0_3))))
(:derived (z_23_4) (or (and (not (v0_4)) (not (c_23_3))) (and (c_23_3) (v0_4))))
(:action feast__excitement__turkey__tuna
  :parameters ()
  :precondition (and (not (v0_4)) (craves_excitement_turkey) (not (of)))
  :effect (and (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of)) (craves_excitement_tuna) (not (craves_excitement_turkey))))
 (:action feast__excitement__turkey__chocolate
  :parameters ()
  :precondition (and (not (v0_4)) (craves_excitement_turkey) (not (of)))
  :effect (and (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of)) (not (craves_excitement_turkey)) (craves_excitement_chocolate)))
 (:action overcome__anxiety__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_anxiety_broccoli) (not (v1_4)) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_anxiety_broccoli)) (fears_anxiety_satisfaction)))
 (:action overcome__boils__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_boils_broccoli) (not (v1_4)) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_boils_broccoli)) (fears_boils_satisfaction)))
 (:action overcome__abrasion__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_abrasion_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_abrasion_excitement) (not (craves_abrasion_turkey))))
 (:action drink__broccoli__shrimp
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (when (z_15_0) (v6_0)) (when (not (z_15_0)) (not (v6_0))) (when (z_15_1) (v6_1)) (when (not (z_15_1)) (not (v6_1))) (when (z_15_2) (v6_2)) (when (not (z_15_2)) (not (v6_2))) (when (z_15_3) (v6_3)) (when (not (z_15_3)) (not (v6_3))) (when (z_15_4) (v6_4)) (when (not (z_15_4)) (not (v6_4))) (when (and (not (v6_4)) (z_15_4)) (of))))
 (:action drink__broccoli__sweetroll
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (when (z_19_0) (v9_0)) (when (not (z_19_0)) (not (v9_0))) (when (z_19_1) (v9_1)) (when (not (z_19_1)) (not (v9_1))) (when (z_19_2) (v9_2)) (when (not (z_19_2)) (not (v9_2))) (when (z_19_3) (v9_3)) (when (not (z_19_3)) (not (v9_3))) (when (z_19_4) (v9_4)) (when (not (z_19_4)) (not (v9_4))) (when (and (not (v9_4)) (z_19_4)) (of))))
 (:action drink__broccoli__scallop
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_20_0) (v10_0)) (when (not (z_20_0)) (not (v10_0))) (when (z_20_1) (v10_1)) (when (not (z_20_1)) (not (v10_1))) (when (z_20_2) (v10_2)) (when (not (z_20_2)) (not (v10_2))) (when (z_20_3) (v10_3)) (when (not (z_20_3)) (not (v10_3))) (when (z_20_4) (v10_4)) (when (not (z_20_4)) (not (v10_4))) (when (and (not (v10_4)) (z_20_4)) (of)) (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of))))
 (:action drink__broccoli__cherry
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (when (z_16_0) (v7_0)) (when (not (z_16_0)) (not (v7_0))) (when (z_16_1) (v7_1)) (when (not (z_16_1)) (not (v7_1))) (when (z_16_2) (v7_2)) (when (not (z_16_2)) (not (v7_2))) (when (z_16_3) (v7_3)) (when (not (z_16_3)) (not (v7_3))) (when (z_16_4) (v7_4)) (when (not (z_16_4)) (not (v7_4))) (when (and (not (v7_4)) (z_16_4)) (of))))
 (:action drink__broccoli__chocolate
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (when (z_12_0) (v4_0)) (when (not (z_12_0)) (not (v4_0))) (when (z_12_1) (v4_1)) (when (not (z_12_1)) (not (v4_1))) (when (z_12_2) (v4_2)) (when (not (z_12_2)) (not (v4_2))) (when (z_12_3) (v4_3)) (when (not (z_12_3)) (not (v4_3))) (when (z_12_4) (v4_4)) (when (not (z_12_4)) (not (v4_4))) (when (and (not (v4_4)) (z_12_4)) (of))))
 (:action drink__broccoli__tuna
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (when (z_14_0) (v5_0)) (when (not (z_14_0)) (not (v5_0))) (when (z_14_1) (v5_1)) (when (not (z_14_1)) (not (v5_1))) (when (z_14_2) (v5_2)) (when (not (z_14_2)) (not (v5_2))) (when (z_14_3) (v5_3)) (when (not (z_14_3)) (not (v5_3))) (when (z_14_4) (v5_4)) (when (not (z_14_4)) (not (v5_4))) (when (and (not (v5_4)) (z_14_4)) (of))))
 (:action drink__broccoli__turkey
  :parameters ()
  :precondition (and (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (when (z_4_0) (v0_0)) (when (not (z_4_0)) (not (v0_0))) (when (z_4_1) (v0_1)) (when (not (z_4_1)) (not (v0_1))) (when (z_4_2) (v0_2)) (when (not (z_4_2)) (not (v0_2))) (when (z_4_3) (v0_3)) (when (not (z_4_3)) (not (v0_3))) (when (z_4_4) (v0_4)) (when (not (z_4_4)) (not (v0_4))) (when (and (not (v0_4)) (z_4_4)) (of))))
 (:action drink__chocolate__cherry
  :parameters ()
  :precondition (and (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (when (z_16_0) (v7_0)) (when (not (z_16_0)) (not (v7_0))) (when (z_16_1) (v7_1)) (when (not (z_16_1)) (not (v7_1))) (when (z_16_2) (v7_2)) (when (not (z_16_2)) (not (v7_2))) (when (z_16_3) (v7_3)) (when (not (z_16_3)) (not (v7_3))) (when (z_16_4) (v7_4)) (when (not (z_16_4)) (not (v7_4))) (when (and (not (v7_4)) (z_16_4)) (of))))
 (:action drink__chocolate__shrimp
  :parameters ()
  :precondition (and (not (v4_4)) (not (of)))
  :effect (and (when (z_15_0) (v6_0)) (when (not (z_15_0)) (not (v6_0))) (when (z_15_1) (v6_1)) (when (not (z_15_1)) (not (v6_1))) (when (z_15_2) (v6_2)) (when (not (z_15_2)) (not (v6_2))) (when (z_15_3) (v6_3)) (when (not (z_15_3)) (not (v6_3))) (when (z_15_4) (v6_4)) (when (not (z_15_4)) (not (v6_4))) (when (and (not (v6_4)) (z_15_4)) (of)) (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of))))
 (:action drink__chocolate__scallop
  :parameters ()
  :precondition (and (not (v4_4)) (not (of)))
  :effect (and (when (z_20_0) (v10_0)) (when (not (z_20_0)) (not (v10_0))) (when (z_20_1) (v10_1)) (when (not (z_20_1)) (not (v10_1))) (when (z_20_2) (v10_2)) (when (not (z_20_2)) (not (v10_2))) (when (z_20_3) (v10_3)) (when (not (z_20_3)) (not (v10_3))) (when (z_20_4) (v10_4)) (when (not (z_20_4)) (not (v10_4))) (when (and (not (v10_4)) (z_20_4)) (of)) (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of))))
 (:action drink__chocolate__turkey
  :parameters ()
  :precondition (and (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (when (z_4_0) (v0_0)) (when (not (z_4_0)) (not (v0_0))) (when (z_4_1) (v0_1)) (when (not (z_4_1)) (not (v0_1))) (when (z_4_2) (v0_2)) (when (not (z_4_2)) (not (v0_2))) (when (z_4_3) (v0_3)) (when (not (z_4_3)) (not (v0_3))) (when (z_4_4) (v0_4)) (when (not (z_4_4)) (not (v0_4))) (when (and (not (v0_4)) (z_4_4)) (of))))
 (:action drink__chocolate__sweetroll
  :parameters ()
  :precondition (and (not (v4_4)) (not (of)))
  :effect (and (when (z_19_0) (v9_0)) (when (not (z_19_0)) (not (v9_0))) (when (z_19_1) (v9_1)) (when (not (z_19_1)) (not (v9_1))) (when (z_19_2) (v9_2)) (when (not (z_19_2)) (not (v9_2))) (when (z_19_3) (v9_3)) (when (not (z_19_3)) (not (v9_3))) (when (z_19_4) (v9_4)) (when (not (z_19_4)) (not (v9_4))) (when (and (not (v9_4)) (z_19_4)) (of)) (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of))))
 (:action drink__chocolate__tuna
  :parameters ()
  :precondition (and (not (v4_4)) (not (of)))
  :effect (and (when (z_14_0) (v5_0)) (when (not (z_14_0)) (not (v5_0))) (when (z_14_1) (v5_1)) (when (not (z_14_1)) (not (v5_1))) (when (z_14_2) (v5_2)) (when (not (z_14_2)) (not (v5_2))) (when (z_14_3) (v5_3)) (when (not (z_14_3)) (not (v5_3))) (when (z_14_4) (v5_4)) (when (not (z_14_4)) (not (v5_4))) (when (and (not (v5_4)) (z_14_4)) (of)) (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of))))
 (:action overcome__sciatica__satisfaction__broccoli
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_broccoli) (craves_sciatica_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_sciatica_broccoli)) (fears_sciatica_satisfaction)))
 (:action drink__chocolate__broccoli
  :parameters ()
  :precondition (and (not (v4_4)) (not (of)))
  :effect (and (when (z_10_0) (v3_0)) (when (not (z_10_0)) (not (v3_0))) (when (z_10_1) (v3_1)) (when (not (z_10_1)) (not (v3_1))) (when (z_10_2) (v3_2)) (when (not (z_10_2)) (not (v3_2))) (when (z_10_3) (v3_3)) (when (not (z_10_3)) (not (v3_3))) (when (z_10_4) (v3_4)) (when (not (z_10_4)) (not (v3_4))) (when (and (not (v3_4)) (z_10_4)) (of)) (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of))))
 (:action drink__turkey__scallop
  :parameters ()
  :precondition (and (not (v0_4)) (not (of)))
  :effect (and (when (z_20_0) (v10_0)) (when (not (z_20_0)) (not (v10_0))) (when (z_20_1) (v10_1)) (when (not (z_20_1)) (not (v10_1))) (when (z_20_2) (v10_2)) (when (not (z_20_2)) (not (v10_2))) (when (z_20_3) (v10_3)) (when (not (z_20_3)) (not (v10_3))) (when (z_20_4) (v10_4)) (when (not (z_20_4)) (not (v10_4))) (when (and (not (v10_4)) (z_20_4)) (of)) (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of))))
 (:action drink__turkey__cherry
  :parameters ()
  :precondition (and (not (v0_4)) (not (of)))
  :effect (and (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of)) (when (z_16_0) (v7_0)) (when (not (z_16_0)) (not (v7_0))) (when (z_16_1) (v7_1)) (when (not (z_16_1)) (not (v7_1))) (when (z_16_2) (v7_2)) (when (not (z_16_2)) (not (v7_2))) (when (z_16_3) (v7_3)) (when (not (z_16_3)) (not (v7_3))) (when (z_16_4) (v7_4)) (when (not (z_16_4)) (not (v7_4))) (when (and (not (v7_4)) (z_16_4)) (of))))
 (:action drink__turkey__tuna
  :parameters ()
  :precondition (and (not (v0_4)) (not (of)))
  :effect (and (when (z_14_0) (v5_0)) (when (not (z_14_0)) (not (v5_0))) (when (z_14_1) (v5_1)) (when (not (z_14_1)) (not (v5_1))) (when (z_14_2) (v5_2)) (when (not (z_14_2)) (not (v5_2))) (when (z_14_3) (v5_3)) (when (not (z_14_3)) (not (v5_3))) (when (z_14_4) (v5_4)) (when (not (z_14_4)) (not (v5_4))) (when (and (not (v5_4)) (z_14_4)) (of)) (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of))))
 (:action drink__turkey__shrimp
  :parameters ()
  :precondition (and (not (v0_4)) (not (of)))
  :effect (and (when (z_15_0) (v6_0)) (when (not (z_15_0)) (not (v6_0))) (when (z_15_1) (v6_1)) (when (not (z_15_1)) (not (v6_1))) (when (z_15_2) (v6_2)) (when (not (z_15_2)) (not (v6_2))) (when (z_15_3) (v6_3)) (when (not (z_15_3)) (not (v6_3))) (when (z_15_4) (v6_4)) (when (not (z_15_4)) (not (v6_4))) (when (and (not (v6_4)) (z_15_4)) (of)) (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of))))
 (:action drink__turkey__sweetroll
  :parameters ()
  :precondition (and (not (v0_4)) (not (of)))
  :effect (and (when (z_19_0) (v9_0)) (when (not (z_19_0)) (not (v9_0))) (when (z_19_1) (v9_1)) (when (not (z_19_1)) (not (v9_1))) (when (z_19_2) (v9_2)) (when (not (z_19_2)) (not (v9_2))) (when (z_19_3) (v9_3)) (when (not (z_19_3)) (not (v9_3))) (when (z_19_4) (v9_4)) (when (not (z_19_4)) (not (v9_4))) (when (and (not (v9_4)) (z_19_4)) (of)) (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of))))
 (:action drink__turkey__chocolate
  :parameters ()
  :precondition (and (not (v0_4)) (not (of)))
  :effect (and (when (z_12_0) (v4_0)) (when (not (z_12_0)) (not (v4_0))) (when (z_12_1) (v4_1)) (when (not (z_12_1)) (not (v4_1))) (when (z_12_2) (v4_2)) (when (not (z_12_2)) (not (v4_2))) (when (z_12_3) (v4_3)) (when (not (z_12_3)) (not (v4_3))) (when (z_12_4) (v4_4)) (when (not (z_12_4)) (not (v4_4))) (when (and (not (v4_4)) (z_12_4)) (of)) (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of))))
 (:action drink__turkey__broccoli
  :parameters ()
  :precondition (and (not (v0_4)) (not (of)))
  :effect (and (when (z_10_0) (v3_0)) (when (not (z_10_0)) (not (v3_0))) (when (z_10_1) (v3_1)) (when (not (z_10_1)) (not (v3_1))) (when (z_10_2) (v3_2)) (when (not (z_10_2)) (not (v3_2))) (when (z_10_3) (v3_3)) (when (not (z_10_3)) (not (v3_3))) (when (z_10_4) (v3_4)) (when (not (z_10_4)) (not (v3_4))) (when (and (not (v3_4)) (z_10_4)) (of)) (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of))))
 (:action drink__tuna__scallop
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_20_0) (v10_0)) (when (not (z_20_0)) (not (v10_0))) (when (z_20_1) (v10_1)) (when (not (z_20_1)) (not (v10_1))) (when (z_20_2) (v10_2)) (when (not (z_20_2)) (not (v10_2))) (when (z_20_3) (v10_3)) (when (not (z_20_3)) (not (v10_3))) (when (z_20_4) (v10_4)) (when (not (z_20_4)) (not (v10_4))) (when (and (not (v10_4)) (z_20_4)) (of)) (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of))))
 (:action drink__tuna__sweetroll
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_19_0) (v9_0)) (when (not (z_19_0)) (not (v9_0))) (when (z_19_1) (v9_1)) (when (not (z_19_1)) (not (v9_1))) (when (z_19_2) (v9_2)) (when (not (z_19_2)) (not (v9_2))) (when (z_19_3) (v9_3)) (when (not (z_19_3)) (not (v9_3))) (when (z_19_4) (v9_4)) (when (not (z_19_4)) (not (v9_4))) (when (and (not (v9_4)) (z_19_4)) (of)) (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of))))
 (:action drink__tuna__cherry
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (when (z_16_0) (v7_0)) (when (not (z_16_0)) (not (v7_0))) (when (z_16_1) (v7_1)) (when (not (z_16_1)) (not (v7_1))) (when (z_16_2) (v7_2)) (when (not (z_16_2)) (not (v7_2))) (when (z_16_3) (v7_3)) (when (not (z_16_3)) (not (v7_3))) (when (z_16_4) (v7_4)) (when (not (z_16_4)) (not (v7_4))) (when (and (not (v7_4)) (z_16_4)) (of))))
 (:action drink__tuna__shrimp
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (when (z_15_0) (v6_0)) (when (not (z_15_0)) (not (v6_0))) (when (z_15_1) (v6_1)) (when (not (z_15_1)) (not (v6_1))) (when (z_15_2) (v6_2)) (when (not (z_15_2)) (not (v6_2))) (when (z_15_3) (v6_3)) (when (not (z_15_3)) (not (v6_3))) (when (z_15_4) (v6_4)) (when (not (z_15_4)) (not (v6_4))) (when (and (not (v6_4)) (z_15_4)) (of))))
 (:action drink__tuna__broccoli
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_10_0) (v3_0)) (when (not (z_10_0)) (not (v3_0))) (when (z_10_1) (v3_1)) (when (not (z_10_1)) (not (v3_1))) (when (z_10_2) (v3_2)) (when (not (z_10_2)) (not (v3_2))) (when (z_10_3) (v3_3)) (when (not (z_10_3)) (not (v3_3))) (when (z_10_4) (v3_4)) (when (not (z_10_4)) (not (v3_4))) (when (and (not (v3_4)) (z_10_4)) (of)) (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of))))
 (:action drink__tuna__turkey
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (when (z_4_0) (v0_0)) (when (not (z_4_0)) (not (v0_0))) (when (z_4_1) (v0_1)) (when (not (z_4_1)) (not (v0_1))) (when (z_4_2) (v0_2)) (when (not (z_4_2)) (not (v0_2))) (when (z_4_3) (v0_3)) (when (not (z_4_3)) (not (v0_3))) (when (z_4_4) (v0_4)) (when (not (z_4_4)) (not (v0_4))) (when (and (not (v0_4)) (z_4_4)) (of))))
 (:action drink__tuna__chocolate
  :parameters ()
  :precondition (and (not (v5_4)) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (when (z_12_0) (v4_0)) (when (not (z_12_0)) (not (v4_0))) (when (z_12_1) (v4_1)) (when (not (z_12_1)) (not (v4_1))) (when (z_12_2) (v4_2)) (when (not (z_12_2)) (not (v4_2))) (when (z_12_3) (v4_3)) (when (not (z_12_3)) (not (v4_3))) (when (z_12_4) (v4_4)) (when (not (z_12_4)) (not (v4_4))) (when (and (not (v4_4)) (z_12_4)) (of))))
 (:action feast__intoxication__tuna__turkey
  :parameters ()
  :precondition (and (not (v5_4)) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (not (craves_intoxication_tuna)) (craves_intoxication_turkey)))
 (:action feast__intoxication__tuna__broccoli
  :parameters ()
  :precondition (and (not (v5_4)) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (not (craves_intoxication_tuna)) (craves_intoxication_broccoli)))
 (:action overcome__grief__satisfaction__broccoli
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_broccoli) (craves_grief_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief_broccoli)) (fears_grief_satisfaction)))
 (:action feast__satisfaction__broccoli__chocolate
  :parameters ()
  :precondition (and (craves_satisfaction_broccoli) (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (not (craves_satisfaction_broccoli)) (craves_satisfaction_chocolate)))
 (:action feast__satisfaction__broccoli__tuna
  :parameters ()
  :precondition (and (craves_satisfaction_broccoli) (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (craves_satisfaction_tuna) (not (craves_satisfaction_broccoli))))
 (:action overcome__depression__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_depression_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_depression_turkey)) (fears_depression_excitement)))
 (:action overcome__prostatitis__excitement__turkey
  :parameters ()
  :precondition (and (craves_prostatitis_turkey) (craves_excitement_turkey) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_prostatitis_turkey)) (fears_prostatitis_excitement)))
 (:action drink__shrimp__broccoli
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_10_0) (v3_0)) (when (not (z_10_0)) (not (v3_0))) (when (z_10_1) (v3_1)) (when (not (z_10_1)) (not (v3_1))) (when (z_10_2) (v3_2)) (when (not (z_10_2)) (not (v3_2))) (when (z_10_3) (v3_3)) (when (not (z_10_3)) (not (v3_3))) (when (z_10_4) (v3_4)) (when (not (z_10_4)) (not (v3_4))) (when (and (not (v3_4)) (z_10_4)) (of)) (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of))))
 (:action drink__shrimp__cherry
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (when (z_16_0) (v7_0)) (when (not (z_16_0)) (not (v7_0))) (when (z_16_1) (v7_1)) (when (not (z_16_1)) (not (v7_1))) (when (z_16_2) (v7_2)) (when (not (z_16_2)) (not (v7_2))) (when (z_16_3) (v7_3)) (when (not (z_16_3)) (not (v7_3))) (when (z_16_4) (v7_4)) (when (not (z_16_4)) (not (v7_4))) (when (and (not (v7_4)) (z_16_4)) (of))))
 (:action drink__shrimp__scallop
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_20_0) (v10_0)) (when (not (z_20_0)) (not (v10_0))) (when (z_20_1) (v10_1)) (when (not (z_20_1)) (not (v10_1))) (when (z_20_2) (v10_2)) (when (not (z_20_2)) (not (v10_2))) (when (z_20_3) (v10_3)) (when (not (z_20_3)) (not (v10_3))) (when (z_20_4) (v10_4)) (when (not (z_20_4)) (not (v10_4))) (when (and (not (v10_4)) (z_20_4)) (of)) (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of))))
 (:action drink__shrimp__turkey
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (when (z_4_0) (v0_0)) (when (not (z_4_0)) (not (v0_0))) (when (z_4_1) (v0_1)) (when (not (z_4_1)) (not (v0_1))) (when (z_4_2) (v0_2)) (when (not (z_4_2)) (not (v0_2))) (when (z_4_3) (v0_3)) (when (not (z_4_3)) (not (v0_3))) (when (z_4_4) (v0_4)) (when (not (z_4_4)) (not (v0_4))) (when (and (not (v0_4)) (z_4_4)) (of))))
 (:action drink__shrimp__chocolate
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (when (z_12_0) (v4_0)) (when (not (z_12_0)) (not (v4_0))) (when (z_12_1) (v4_1)) (when (not (z_12_1)) (not (v4_1))) (when (z_12_2) (v4_2)) (when (not (z_12_2)) (not (v4_2))) (when (z_12_3) (v4_3)) (when (not (z_12_3)) (not (v4_3))) (when (z_12_4) (v4_4)) (when (not (z_12_4)) (not (v4_4))) (when (and (not (v4_4)) (z_12_4)) (of))))
 (:action drink__shrimp__sweetroll
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_19_0) (v9_0)) (when (not (z_19_0)) (not (v9_0))) (when (z_19_1) (v9_1)) (when (not (z_19_1)) (not (v9_1))) (when (z_19_2) (v9_2)) (when (not (z_19_2)) (not (v9_2))) (when (z_19_3) (v9_3)) (when (not (z_19_3)) (not (v9_3))) (when (z_19_4) (v9_4)) (when (not (z_19_4)) (not (v9_4))) (when (and (not (v9_4)) (z_19_4)) (of)) (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of))))
 (:action drink__shrimp__tuna
  :parameters ()
  :precondition (and (not (v6_4)) (not (of)))
  :effect (and (when (z_14_0) (v5_0)) (when (not (z_14_0)) (not (v5_0))) (when (z_14_1) (v5_1)) (when (not (z_14_1)) (not (v5_1))) (when (z_14_2) (v5_2)) (when (not (z_14_2)) (not (v5_2))) (when (z_14_3) (v5_3)) (when (not (z_14_3)) (not (v5_3))) (when (z_14_4) (v5_4)) (when (not (z_14_4)) (not (v5_4))) (when (and (not (v5_4)) (z_14_4)) (of)) (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of))))
 (:action drink__cherry__chocolate
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_12_0) (v4_0)) (when (not (z_12_0)) (not (v4_0))) (when (z_12_1) (v4_1)) (when (not (z_12_1)) (not (v4_1))) (when (z_12_2) (v4_2)) (when (not (z_12_2)) (not (v4_2))) (when (z_12_3) (v4_3)) (when (not (z_12_3)) (not (v4_3))) (when (z_12_4) (v4_4)) (when (not (z_12_4)) (not (v4_4))) (when (and (not (v4_4)) (z_12_4)) (of)) (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of))))
 (:action drink__cherry__broccoli
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_10_0) (v3_0)) (when (not (z_10_0)) (not (v3_0))) (when (z_10_1) (v3_1)) (when (not (z_10_1)) (not (v3_1))) (when (z_10_2) (v3_2)) (when (not (z_10_2)) (not (v3_2))) (when (z_10_3) (v3_3)) (when (not (z_10_3)) (not (v3_3))) (when (z_10_4) (v3_4)) (when (not (z_10_4)) (not (v3_4))) (when (and (not (v3_4)) (z_10_4)) (of)) (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of))))
 (:action drink__cherry__scallop
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_20_0) (v10_0)) (when (not (z_20_0)) (not (v10_0))) (when (z_20_1) (v10_1)) (when (not (z_20_1)) (not (v10_1))) (when (z_20_2) (v10_2)) (when (not (z_20_2)) (not (v10_2))) (when (z_20_3) (v10_3)) (when (not (z_20_3)) (not (v10_3))) (when (z_20_4) (v10_4)) (when (not (z_20_4)) (not (v10_4))) (when (and (not (v10_4)) (z_20_4)) (of)) (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of))))
 (:action drink__cherry__tuna
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_14_0) (v5_0)) (when (not (z_14_0)) (not (v5_0))) (when (z_14_1) (v5_1)) (when (not (z_14_1)) (not (v5_1))) (when (z_14_2) (v5_2)) (when (not (z_14_2)) (not (v5_2))) (when (z_14_3) (v5_3)) (when (not (z_14_3)) (not (v5_3))) (when (z_14_4) (v5_4)) (when (not (z_14_4)) (not (v5_4))) (when (and (not (v5_4)) (z_14_4)) (of)) (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of))))
 (:action drink__cherry__turkey
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_4_0) (v0_0)) (when (not (z_4_0)) (not (v0_0))) (when (z_4_1) (v0_1)) (when (not (z_4_1)) (not (v0_1))) (when (z_4_2) (v0_2)) (when (not (z_4_2)) (not (v0_2))) (when (z_4_3) (v0_3)) (when (not (z_4_3)) (not (v0_3))) (when (z_4_4) (v0_4)) (when (not (z_4_4)) (not (v0_4))) (when (and (not (v0_4)) (z_4_4)) (of)) (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of))))
 (:action drink__cherry__shrimp
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_15_0) (v6_0)) (when (not (z_15_0)) (not (v6_0))) (when (z_15_1) (v6_1)) (when (not (z_15_1)) (not (v6_1))) (when (z_15_2) (v6_2)) (when (not (z_15_2)) (not (v6_2))) (when (z_15_3) (v6_3)) (when (not (z_15_3)) (not (v6_3))) (when (z_15_4) (v6_4)) (when (not (z_15_4)) (not (v6_4))) (when (and (not (v6_4)) (z_15_4)) (of)) (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of))))
 (:action drink__cherry__sweetroll
  :parameters ()
  :precondition (and (not (v7_4)) (not (of)))
  :effect (and (when (z_19_0) (v9_0)) (when (not (z_19_0)) (not (v9_0))) (when (z_19_1) (v9_1)) (when (not (z_19_1)) (not (v9_1))) (when (z_19_2) (v9_2)) (when (not (z_19_2)) (not (v9_2))) (when (z_19_3) (v9_3)) (when (not (z_19_3)) (not (v9_3))) (when (z_19_4) (v9_4)) (when (not (z_19_4)) (not (v9_4))) (when (and (not (v9_4)) (z_19_4)) (of)) (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of))))
 (:action succumb__prostatitis__excitement__turkey
  :parameters ()
  :precondition (and (fears_prostatitis_excitement) (craves_excitement_turkey) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_prostatitis_turkey) (not (fears_prostatitis_excitement))))
 (:action succumb__prostatitis__excitement__chocolate
  :parameters ()
  :precondition (and (fears_prostatitis_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_prostatitis_chocolate) (not (fears_prostatitis_excitement))))
 (:action succumb__prostatitis__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_prostatitis_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_prostatitis_tuna) (not (fears_prostatitis_excitement))))
 (:action overcome__anxiety__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_anxiety_broccoli) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anxiety_broccoli)) (fears_anxiety_intoxication)))
 (:action succumb__depression__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (fears_depression_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_depression_turkey) (not (fears_depression_excitement))))
 (:action succumb__depression__excitement__chocolate
  :parameters ()
  :precondition (and (fears_depression_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_depression_chocolate) (not (fears_depression_excitement))))
 (:action succumb__depression__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_depression_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_depression_excitement)) (craves_depression_tuna)))
 (:action feast__excitement__chocolate__turkey
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (craves_excitement_turkey) (not (craves_excitement_chocolate))))
 (:action feast__excitement__chocolate__broccoli
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_excitement_chocolate)) (craves_excitement_broccoli)))
 (:action feast__excitement__chocolate__shrimp
  :parameters ()
  :precondition (and (not (v4_4)) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_excitement_chocolate)) (craves_excitement_shrimp)))
 (:action feast__lubricity__sweetroll__scallop
  :parameters ()
  :precondition (and (craves_lubricity_sweetroll) (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (not (craves_lubricity_sweetroll)) (craves_lubricity_scallop)))
 (:action feast__lubricity__sweetroll__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_sweetroll) (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (not (craves_lubricity_sweetroll)) (craves_lubricity_shrimp)))
 (:action overcome__boils__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_boils_broccoli) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_boils_intoxication) (not (craves_boils_broccoli))))
 (:action feast__excitement__tuna__broccoli
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (not (craves_excitement_tuna)) (craves_excitement_broccoli)))
 (:action feast__excitement__tuna__turkey
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v5_4)) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (craves_excitement_turkey) (not (craves_excitement_tuna))))
 (:action overcome__abrasion__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_abrasion_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_abrasion_intoxication) (not (craves_abrasion_turkey))))
 (:action succumb__grief__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_grief_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief_tuna) (not (fears_grief_satisfaction))))
 (:action succumb__grief__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_grief_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_grief_satisfaction)) (craves_grief_chocolate)))
 (:action succumb__grief__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_grief_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief_broccoli) (not (fears_grief_satisfaction))))
 (:action succumb__sciatica__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_sciatica_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_sciatica_chocolate) (not (fears_sciatica_satisfaction))))
 (:action succumb__sciatica__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_sciatica_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_sciatica_broccoli) (not (fears_sciatica_satisfaction))))
 (:action succumb__sciatica__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_sciatica_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_sciatica_satisfaction)) (craves_sciatica_tuna)))
 (:action feast__intoxication__broccoli__tuna
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (not (craves_intoxication_broccoli)) (craves_intoxication_tuna)))
 (:action feast__intoxication__broccoli__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (not (craves_intoxication_broccoli)) (craves_intoxication_chocolate)))
 (:action overcome__sciatica__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_broccoli) (craves_sciatica_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_sciatica_intoxication) (not (craves_sciatica_broccoli))))
 (:action feast__intoxication__turkey__chocolate
  :parameters ()
  :precondition (and (not (v0_4)) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of)) (not (craves_intoxication_turkey)) (craves_intoxication_chocolate)))
 (:action feast__intoxication__turkey__tuna
  :parameters ()
  :precondition (and (not (v0_4)) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of)) (not (craves_intoxication_turkey)) (craves_intoxication_tuna)))
 (:action succumb__abrasion__excitement__chocolate
  :parameters ()
  :precondition (and (fears_abrasion_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_abrasion_excitement)) (craves_abrasion_chocolate)))
 (:action succumb__abrasion__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_abrasion_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_abrasion_excitement)) (craves_abrasion_tuna)))
 (:action succumb__abrasion__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (fears_abrasion_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_abrasion_excitement)) (craves_abrasion_turkey)))
 (:action feast__satisfaction__chocolate__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_chocolate) (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_satisfaction_chocolate)) (craves_satisfaction_turkey)))
 (:action feast__satisfaction__chocolate__broccoli
  :parameters ()
  :precondition (and (craves_satisfaction_chocolate) (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_satisfaction_chocolate)) (craves_satisfaction_broccoli)))
 (:action feast__satisfaction__chocolate__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_chocolate) (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_satisfaction_chocolate)) (craves_satisfaction_shrimp)))
 (:action overcome__grief__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_broccoli) (craves_grief_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_grief_intoxication) (not (craves_grief_broccoli))))
 (:action feast__satisfaction__tuna__broccoli
  :parameters ()
  :precondition (and (not (v5_4)) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (not (craves_satisfaction_tuna)) (craves_satisfaction_broccoli)))
 (:action feast__satisfaction__tuna__turkey
  :parameters ()
  :precondition (and (not (v5_4)) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (not (craves_satisfaction_tuna)) (craves_satisfaction_turkey)))
 (:action overcome__depression__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_depression_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_depression_turkey)) (fears_depression_intoxication)))
 (:action drink__sweetroll__shrimp
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (when (z_15_0) (v6_0)) (when (not (z_15_0)) (not (v6_0))) (when (z_15_1) (v6_1)) (when (not (z_15_1)) (not (v6_1))) (when (z_15_2) (v6_2)) (when (not (z_15_2)) (not (v6_2))) (when (z_15_3) (v6_3)) (when (not (z_15_3)) (not (v6_3))) (when (z_15_4) (v6_4)) (when (not (z_15_4)) (not (v6_4))) (when (and (not (v6_4)) (z_15_4)) (of))))
 (:action drink__sweetroll__scallop
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_20_0) (v10_0)) (when (not (z_20_0)) (not (v10_0))) (when (z_20_1) (v10_1)) (when (not (z_20_1)) (not (v10_1))) (when (z_20_2) (v10_2)) (when (not (z_20_2)) (not (v10_2))) (when (z_20_3) (v10_3)) (when (not (z_20_3)) (not (v10_3))) (when (z_20_4) (v10_4)) (when (not (z_20_4)) (not (v10_4))) (when (and (not (v10_4)) (z_20_4)) (of)) (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of))))
 (:action drink__sweetroll__cherry
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (when (z_16_0) (v7_0)) (when (not (z_16_0)) (not (v7_0))) (when (z_16_1) (v7_1)) (when (not (z_16_1)) (not (v7_1))) (when (z_16_2) (v7_2)) (when (not (z_16_2)) (not (v7_2))) (when (z_16_3) (v7_3)) (when (not (z_16_3)) (not (v7_3))) (when (z_16_4) (v7_4)) (when (not (z_16_4)) (not (v7_4))) (when (and (not (v7_4)) (z_16_4)) (of))))
 (:action drink__sweetroll__chocolate
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (when (z_12_0) (v4_0)) (when (not (z_12_0)) (not (v4_0))) (when (z_12_1) (v4_1)) (when (not (z_12_1)) (not (v4_1))) (when (z_12_2) (v4_2)) (when (not (z_12_2)) (not (v4_2))) (when (z_12_3) (v4_3)) (when (not (z_12_3)) (not (v4_3))) (when (z_12_4) (v4_4)) (when (not (z_12_4)) (not (v4_4))) (when (and (not (v4_4)) (z_12_4)) (of))))
 (:action drink__sweetroll__broccoli
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_10_0) (v3_0)) (when (not (z_10_0)) (not (v3_0))) (when (z_10_1) (v3_1)) (when (not (z_10_1)) (not (v3_1))) (when (z_10_2) (v3_2)) (when (not (z_10_2)) (not (v3_2))) (when (z_10_3) (v3_3)) (when (not (z_10_3)) (not (v3_3))) (when (z_10_4) (v3_4)) (when (not (z_10_4)) (not (v3_4))) (when (and (not (v3_4)) (z_10_4)) (of)) (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of))))
 (:action drink__sweetroll__tuna
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (when (z_14_0) (v5_0)) (when (not (z_14_0)) (not (v5_0))) (when (z_14_1) (v5_1)) (when (not (z_14_1)) (not (v5_1))) (when (z_14_2) (v5_2)) (when (not (z_14_2)) (not (v5_2))) (when (z_14_3) (v5_3)) (when (not (z_14_3)) (not (v5_3))) (when (z_14_4) (v5_4)) (when (not (z_14_4)) (not (v5_4))) (when (and (not (v5_4)) (z_14_4)) (of))))
 (:action drink__sweetroll__turkey
  :parameters ()
  :precondition (and (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (when (z_4_0) (v0_0)) (when (not (z_4_0)) (not (v0_0))) (when (z_4_1) (v0_1)) (when (not (z_4_1)) (not (v0_1))) (when (z_4_2) (v0_2)) (when (not (z_4_2)) (not (v0_2))) (when (z_4_3) (v0_3)) (when (not (z_4_3)) (not (v0_3))) (when (z_4_4) (v0_4)) (when (not (z_4_4)) (not (v0_4))) (when (and (not (v0_4)) (z_4_4)) (of))))
 (:action drink__scallop__broccoli
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (when (z_10_0) (v3_0)) (when (not (z_10_0)) (not (v3_0))) (when (z_10_1) (v3_1)) (when (not (z_10_1)) (not (v3_1))) (when (z_10_2) (v3_2)) (when (not (z_10_2)) (not (v3_2))) (when (z_10_3) (v3_3)) (when (not (z_10_3)) (not (v3_3))) (when (z_10_4) (v3_4)) (when (not (z_10_4)) (not (v3_4))) (when (and (not (v3_4)) (z_10_4)) (of))))
 (:action drink__scallop__turkey
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (when (z_4_0) (v0_0)) (when (not (z_4_0)) (not (v0_0))) (when (z_4_1) (v0_1)) (when (not (z_4_1)) (not (v0_1))) (when (z_4_2) (v0_2)) (when (not (z_4_2)) (not (v0_2))) (when (z_4_3) (v0_3)) (when (not (z_4_3)) (not (v0_3))) (when (z_4_4) (v0_4)) (when (not (z_4_4)) (not (v0_4))) (when (and (not (v0_4)) (z_4_4)) (of))))
 (:action drink__scallop__chocolate
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (when (z_12_0) (v4_0)) (when (not (z_12_0)) (not (v4_0))) (when (z_12_1) (v4_1)) (when (not (z_12_1)) (not (v4_1))) (when (z_12_2) (v4_2)) (when (not (z_12_2)) (not (v4_2))) (when (z_12_3) (v4_3)) (when (not (z_12_3)) (not (v4_3))) (when (z_12_4) (v4_4)) (when (not (z_12_4)) (not (v4_4))) (when (and (not (v4_4)) (z_12_4)) (of))))
 (:action overcome__prostatitis__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_prostatitis_turkey) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_prostatitis_intoxication) (not (craves_prostatitis_turkey))))
 (:action drink__scallop__sweetroll
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (when (z_19_0) (v9_0)) (when (not (z_19_0)) (not (v9_0))) (when (z_19_1) (v9_1)) (when (not (z_19_1)) (not (v9_1))) (when (z_19_2) (v9_2)) (when (not (z_19_2)) (not (v9_2))) (when (z_19_3) (v9_3)) (when (not (z_19_3)) (not (v9_3))) (when (z_19_4) (v9_4)) (when (not (z_19_4)) (not (v9_4))) (when (and (not (v9_4)) (z_19_4)) (of))))
 (:action drink__scallop__tuna
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (when (z_14_0) (v5_0)) (when (not (z_14_0)) (not (v5_0))) (when (z_14_1) (v5_1)) (when (not (z_14_1)) (not (v5_1))) (when (z_14_2) (v5_2)) (when (not (z_14_2)) (not (v5_2))) (when (z_14_3) (v5_3)) (when (not (z_14_3)) (not (v5_3))) (when (z_14_4) (v5_4)) (when (not (z_14_4)) (not (v5_4))) (when (and (not (v5_4)) (z_14_4)) (of))))
 (:action drink__scallop__cherry
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (when (z_16_0) (v7_0)) (when (not (z_16_0)) (not (v7_0))) (when (z_16_1) (v7_1)) (when (not (z_16_1)) (not (v7_1))) (when (z_16_2) (v7_2)) (when (not (z_16_2)) (not (v7_2))) (when (z_16_3) (v7_3)) (when (not (z_16_3)) (not (v7_3))) (when (z_16_4) (v7_4)) (when (not (z_16_4)) (not (v7_4))) (when (and (not (v7_4)) (z_16_4)) (of))))
 (:action drink__scallop__shrimp
  :parameters ()
  :precondition (and (not (v10_4)) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (when (z_15_0) (v6_0)) (when (not (z_15_0)) (not (v6_0))) (when (z_15_1) (v6_1)) (when (not (z_15_1)) (not (v6_1))) (when (z_15_2) (v6_2)) (when (not (z_15_2)) (not (v6_2))) (when (z_15_3) (v6_3)) (when (not (z_15_3)) (not (v6_3))) (when (z_15_4) (v6_4)) (when (not (z_15_4)) (not (v6_4))) (when (and (not (v6_4)) (z_15_4)) (of))))
 (:action succumb__boils__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_boils_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_boils_satisfaction)) (craves_boils_chocolate)))
 (:action succumb__boils__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_boils_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_boils_tuna) (not (fears_boils_satisfaction))))
 (:action succumb__boils__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_satisfaction_broccoli) (fears_boils_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_boils_broccoli) (not (fears_boils_satisfaction))))
 (:action succumb__anxiety__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_anxiety_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_anxiety_satisfaction)) (craves_anxiety_tuna)))
 (:action succumb__anxiety__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_anxiety_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_anxiety_satisfaction)) (craves_anxiety_broccoli)))
 (:action succumb__anxiety__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_anxiety_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_anxiety_satisfaction)) (craves_anxiety_chocolate)))
 (:action succumb__prostatitis__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_prostatitis_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_prostatitis_broccoli) (not (fears_prostatitis_excitement))))
 (:action succumb__prostatitis__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_prostatitis_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_prostatitis_shrimp) (not (fears_prostatitis_excitement))))
 (:action overcome__anxiety__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_anxiety_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_anxiety_intoxication) (not (craves_anxiety_chocolate))))
 (:action overcome__anxiety__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_anxiety_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anxiety_tuna)) (fears_anxiety_intoxication)))
 (:action succumb__depression__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_depression_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_depression_excitement)) (craves_depression_shrimp)))
 (:action feast__lubricity__shrimp__sweetroll
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (craves_lubricity_sweetroll) (not (craves_lubricity_shrimp))))
 (:action feast__lubricity__shrimp__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (craves_lubricity_chocolate) (not (craves_lubricity_shrimp))))
 (:action feast__lubricity__shrimp__cherry
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (craves_lubricity_cherry) (not (craves_lubricity_shrimp))))
 (:action succumb__depression__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_depression_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_depression_broccoli) (not (fears_depression_excitement))))
 (:action feast__excitement__broccoli__chocolate
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (craves_excitement_chocolate) (not (craves_excitement_broccoli))))
 (:action feast__excitement__broccoli__tuna
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (craves_excitement_tuna) (not (craves_excitement_broccoli))))
 (:action overcome__boils__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_boils_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_boils_intoxication) (not (craves_boils_tuna))))
 (:action overcome__boils__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_boils_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_boils_intoxication) (not (craves_boils_chocolate))))
 (:action feast__lubricity__scallop__cherry
  :parameters ()
  :precondition (and (not (v10_4)) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (craves_lubricity_cherry) (not (craves_lubricity_scallop))))
 (:action feast__lubricity__scallop__sweetroll
  :parameters ()
  :precondition (and (not (v10_4)) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (craves_lubricity_sweetroll) (not (craves_lubricity_scallop))))
 (:action overcome__anxiety__excitement__broccoli
  :parameters ()
  :precondition (and (craves_anxiety_broccoli) (craves_excitement_broccoli) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_anxiety_broccoli)) (fears_anxiety_excitement)))
 (:action overcome__anxiety__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_anxiety_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anxiety_excitement) (not (craves_anxiety_chocolate))))
 (:action overcome__anxiety__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_anxiety_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anxiety_excitement) (not (craves_anxiety_tuna))))
 (:action succumb__prostatitis__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_prostatitis_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_prostatitis_intoxication)) (craves_prostatitis_chocolate)))
 (:action succumb__prostatitis__intoxication__broccoli
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_prostatitis_broccoli) (not (fears_prostatitis_intoxication))))
 (:action succumb__prostatitis__intoxication__tuna
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_prostatitis_tuna) (not (fears_prostatitis_intoxication))))
 (:action succumb__prostatitis__intoxication__turkey
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_prostatitis_turkey) (not (fears_prostatitis_intoxication))))
 (:action overcome__anxiety__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_anxiety_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_anxiety_satisfaction) (not (craves_anxiety_tuna))))
 (:action overcome__anxiety__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_anxiety_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_anxiety_satisfaction) (not (craves_anxiety_chocolate))))
 (:action feast__excitement__shrimp__chocolate
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (not (craves_excitement_shrimp)) (craves_excitement_chocolate)))
 (:action feast__excitement__shrimp__cherry
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (not (craves_excitement_shrimp)) (craves_excitement_cherry)))
 (:action feast__excitement__shrimp__sweetroll
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (not (craves_excitement_shrimp)) (craves_excitement_sweetroll)))
 (:action overcome__abrasion__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_abrasion_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_abrasion_intoxication) (not (craves_abrasion_chocolate))))
 (:action overcome__abrasion__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_abrasion_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_abrasion_intoxication) (not (craves_abrasion_tuna))))
 (:action overcome__boils__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_boils_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_boils_excitement) (not (craves_boils_tuna))))
 (:action overcome__boils__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (craves_boils_broccoli) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_boils_excitement) (not (craves_boils_broccoli))))
 (:action overcome__boils__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_boils_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_boils_excitement) (not (craves_boils_chocolate))))
 (:action overcome__boils__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_boils_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_boils_satisfaction) (not (craves_boils_chocolate))))
 (:action overcome__boils__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_boils_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_boils_tuna)) (fears_boils_satisfaction)))
 (:action overcome__angina__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_shrimp) (craves_angina_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_angina_lubricity) (not (craves_angina_shrimp))))
 (:action overcome__abrasion__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_abrasion_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_abrasion_excitement) (not (craves_abrasion_chocolate))))
 (:action overcome__abrasion__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_abrasion_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_abrasion_excitement) (not (craves_abrasion_tuna))))
 (:action overcome__abrasion__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_abrasion_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_abrasion_chocolate)) (fears_abrasion_satisfaction)))
 (:action overcome__abrasion__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_abrasion_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_abrasion_turkey)) (fears_abrasion_satisfaction)))
 (:action overcome__abrasion__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_abrasion_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_abrasion_satisfaction) (not (craves_abrasion_tuna))))
 (:action overcome__laceration__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_laceration_shrimp) (not (v11_4)) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_laceration_shrimp)) (fears_laceration_lubricity)))
 (:action overcome__angina__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v2_4)) (craves_angina_shrimp) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_angina_excitement) (not (craves_angina_shrimp))))
 (:action overcome__angina__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v1_4)) (craves_angina_shrimp) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_angina_satisfaction) (not (craves_angina_shrimp))))
 (:action succumb__sciatica__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_sciatica_chocolate) (not (fears_sciatica_intoxication))))
 (:action succumb__sciatica__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_sciatica_broccoli) (not (fears_sciatica_intoxication))))
 (:action succumb__sciatica__intoxication__tuna
  :parameters ()
  :precondition (and (fears_sciatica_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_sciatica_intoxication)) (craves_sciatica_tuna)))
 (:action succumb__sciatica__intoxication__turkey
  :parameters ()
  :precondition (and (craves_intoxication_turkey) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_sciatica_intoxication)) (craves_sciatica_turkey)))
 (:action overcome__grief2__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_scallop) (craves_grief2_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief2_lubricity) (not (craves_grief2_scallop))))
 (:action overcome__laceration__excitement__shrimp
  :parameters ()
  :precondition (and (craves_laceration_shrimp) (craves_excitement_shrimp) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_laceration_excitement) (not (craves_laceration_shrimp))))
 (:action overcome__laceration__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (craves_laceration_shrimp) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_laceration_shrimp)) (fears_laceration_satisfaction)))
 (:action overcome__loneliness__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_loneliness_scallop) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_loneliness_scallop)) (fears_loneliness_lubricity)))
 (:action succumb__grief__satisfaction__turkey
  :parameters ()
  :precondition (and (fears_grief_satisfaction) (craves_satisfaction_turkey) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief_turkey) (not (fears_grief_satisfaction))))
 (:action succumb__grief__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_grief_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief_shrimp) (not (fears_grief_satisfaction))))
 (:action succumb__depression__intoxication__tuna
  :parameters ()
  :precondition (and (fears_depression_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_depression_intoxication)) (craves_depression_tuna)))
 (:action succumb__depression__intoxication__turkey
  :parameters ()
  :precondition (and (craves_intoxication_turkey) (fears_depression_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_depression_intoxication)) (craves_depression_turkey)))
 (:action succumb__depression__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_depression_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_depression_intoxication)) (craves_depression_chocolate)))
 (:action succumb__depression__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (fears_depression_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_depression_intoxication)) (craves_depression_broccoli)))
 (:action succumb__sciatica__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_sciatica_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_sciatica_shrimp) (not (fears_sciatica_satisfaction))))
 (:action succumb__sciatica__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_sciatica_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_sciatica_satisfaction)) (craves_sciatica_turkey)))
 (:action succumb__grief__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_grief_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_grief_intoxication)) (craves_grief_chocolate)))
 (:action succumb__grief__intoxication__broccoli
  :parameters ()
  :precondition (and (fears_grief_intoxication) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief_broccoli) (not (fears_grief_intoxication))))
 (:action succumb__grief__intoxication__tuna
  :parameters ()
  :precondition (and (fears_grief_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_grief_intoxication)) (craves_grief_tuna)))
 (:action succumb__grief__intoxication__turkey
  :parameters ()
  :precondition (and (fears_grief_intoxication) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief_turkey) (not (fears_grief_intoxication))))
 (:action overcome__sciatica__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v2_4)) (craves_sciatica_broccoli) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_broccoli))))
 (:action overcome__sciatica__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_sciatica_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_chocolate))))
 (:action overcome__sciatica__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_sciatica_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_tuna))))
 (:action overcome__sciatica__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_sciatica_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_sciatica_satisfaction) (not (craves_sciatica_chocolate))))
 (:action overcome__sciatica__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_sciatica_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_sciatica_satisfaction) (not (craves_sciatica_tuna))))
 (:action feast__intoxication__chocolate__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_intoxication_chocolate)) (craves_intoxication_broccoli)))
 (:action feast__intoxication__chocolate__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_intoxication_chocolate)) (craves_intoxication_shrimp)))
 (:action feast__intoxication__chocolate__turkey
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_intoxication_chocolate)) (craves_intoxication_turkey)))
 (:action overcome__sciatica__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_sciatica_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_sciatica_intoxication) (not (craves_sciatica_chocolate))))
 (:action overcome__sciatica__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_sciatica_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_sciatica_intoxication) (not (craves_sciatica_tuna))))
 (:action overcome__grief__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_grief_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief_excitement) (not (craves_grief_tuna))))
 (:action overcome__grief__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v2_4)) (craves_grief_broccoli) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief_excitement) (not (craves_grief_broccoli))))
 (:action overcome__grief__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_grief_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_grief_chocolate)) (fears_grief_excitement)))
 (:action succumb__abrasion__intoxication__turkey
  :parameters ()
  :precondition (and (craves_intoxication_turkey) (fears_abrasion_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_abrasion_intoxication)) (craves_abrasion_turkey)))
 (:action succumb__abrasion__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_abrasion_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_abrasion_intoxication)) (craves_abrasion_chocolate)))
 (:action succumb__abrasion__intoxication__tuna
  :parameters ()
  :precondition (and (fears_abrasion_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_abrasion_intoxication)) (craves_abrasion_tuna)))
 (:action succumb__abrasion__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (fears_abrasion_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_abrasion_broccoli) (not (fears_abrasion_intoxication))))
 (:action overcome__grief__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_grief_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_grief_satisfaction) (not (craves_grief_tuna))))
 (:action overcome__grief__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_grief_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief_chocolate)) (fears_grief_satisfaction)))
 (:action succumb__abrasion__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_abrasion_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_abrasion_broccoli) (not (fears_abrasion_excitement))))
 (:action succumb__abrasion__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_abrasion_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_abrasion_excitement)) (craves_abrasion_shrimp)))
 (:action overcome__grief__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_grief_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_grief_chocolate)) (fears_grief_intoxication)))
 (:action overcome__grief__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_grief_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_grief_intoxication) (not (craves_grief_tuna))))
 (:action overcome__depression__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_depression_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_depression_chocolate)) (fears_depression_excitement)))
 (:action overcome__depression__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_depression_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_depression_excitement) (not (craves_depression_tuna))))
 (:action feast__satisfaction__turkey__tuna
  :parameters ()
  :precondition (and (not (v0_4)) (craves_satisfaction_turkey) (not (of)))
  :effect (and (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of)) (craves_satisfaction_tuna) (not (craves_satisfaction_turkey))))
 (:action feast__satisfaction__turkey__chocolate
  :parameters ()
  :precondition (and (not (v0_4)) (craves_satisfaction_turkey) (not (of)))
  :effect (and (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of)) (not (craves_satisfaction_turkey)) (craves_satisfaction_chocolate)))
 (:action overcome__depression__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_depression_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_depression_chocolate)) (fears_depression_satisfaction)))
 (:action overcome__depression__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_depression_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_depression_turkey)) (fears_depression_satisfaction)))
 (:action overcome__depression__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_depression_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_depression_satisfaction) (not (craves_depression_tuna))))
 (:action feast__satisfaction__shrimp__chocolate
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (not (craves_satisfaction_shrimp)) (craves_satisfaction_chocolate)))
 (:action feast__satisfaction__shrimp__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (craves_satisfaction_cherry) (not (craves_satisfaction_shrimp))))
 (:action feast__satisfaction__shrimp__sweetroll
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (craves_satisfaction_sweetroll) (not (craves_satisfaction_shrimp))))
 (:action overcome__depression__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_depression_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_depression_tuna)) (fears_depression_intoxication)))
 (:action overcome__depression__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_depression_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_depression_chocolate)) (fears_depression_intoxication)))
 (:action overcome__prostatitis__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_prostatitis_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_prostatitis_chocolate)) (fears_prostatitis_excitement)))
 (:action overcome__prostatitis__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_prostatitis_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_prostatitis_tuna)) (fears_prostatitis_excitement)))
 (:action overcome__prostatitis__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_prostatitis_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_prostatitis_satisfaction) (not (craves_prostatitis_chocolate))))
 (:action overcome__prostatitis__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_prostatitis_turkey) (craves_satisfaction_turkey) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_prostatitis_satisfaction) (not (craves_prostatitis_turkey))))
 (:action overcome__prostatitis__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_prostatitis_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_prostatitis_satisfaction) (not (craves_prostatitis_tuna))))
 (:action overcome__prostatitis__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_prostatitis_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_prostatitis_chocolate)) (fears_prostatitis_intoxication)))
 (:action overcome__prostatitis__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_prostatitis_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_prostatitis_tuna)) (fears_prostatitis_intoxication)))
 (:action overcome__jealousy__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v2_4)) (craves_jealousy_shrimp) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_jealousy_shrimp)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v1_4)) (craves_jealousy_shrimp) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_jealousy_satisfaction) (not (craves_jealousy_shrimp))))
 (:action overcome__jealousy__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_shrimp) (craves_jealousy_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_jealousy_lubricity) (not (craves_jealousy_shrimp))))
 (:action succumb__boils__intoxication__tuna
  :parameters ()
  :precondition (and (fears_boils_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_boils_tuna) (not (fears_boils_intoxication))))
 (:action succumb__boils__intoxication__broccoli
  :parameters ()
  :precondition (and (fears_boils_intoxication) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_broccoli)))
 (:action succumb__boils__intoxication__turkey
  :parameters ()
  :precondition (and (fears_boils_intoxication) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_turkey)))
 (:action succumb__boils__intoxication__chocolate
  :parameters ()
  :precondition (and (fears_boils_intoxication) (craves_intoxication_chocolate) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_chocolate)))
 (:action overcome__dread__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_scallop) (craves_dread_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_dread_scallop)) (fears_dread_lubricity)))
 (:action succumb__boils__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_boils_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_boils_satisfaction)) (craves_boils_turkey)))
 (:action succumb__boils__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_boils_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_boils_satisfaction)) (craves_boils_shrimp)))
 (:action succumb__anxiety__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (fears_anxiety_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_broccoli)))
 (:action succumb__anxiety__intoxication__turkey
  :parameters ()
  :precondition (and (craves_intoxication_turkey) (fears_anxiety_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_turkey)))
 (:action succumb__anxiety__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_anxiety_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_chocolate)))
 (:action succumb__anxiety__intoxication__tuna
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_tuna)))
 (:action overcome__hangover__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_hangover_scallop) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_hangover_scallop)) (fears_hangover_lubricity)))
 (:action succumb__anxiety__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_anxiety_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anxiety_shrimp) (not (fears_anxiety_satisfaction))))
 (:action succumb__anxiety__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_anxiety_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anxiety_turkey) (not (fears_anxiety_satisfaction))))
 (:action succumb__prostatitis__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_prostatitis_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_prostatitis_sweetroll) (not (fears_prostatitis_excitement))))
 (:action succumb__prostatitis__excitement__cherry
  :parameters ()
  :precondition (and (fears_prostatitis_excitement) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_prostatitis_cherry) (not (fears_prostatitis_excitement))))
 (:action feast__lubricity__chocolate__broccoli
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_lubricity_chocolate)) (craves_lubricity_broccoli)))
 (:action feast__lubricity__chocolate__turkey
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (craves_lubricity_turkey) (not (craves_lubricity_chocolate))))
 (:action succumb__prostatitis__satisfaction__cherry
  :parameters ()
  :precondition (and (fears_prostatitis_satisfaction) (craves_satisfaction_cherry) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_prostatitis_satisfaction)) (craves_prostatitis_cherry)))
 (:action overcome__anxiety__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_anxiety_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anxiety_lubricity) (not (craves_anxiety_chocolate))))
 (:action overcome__anxiety__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_anxiety_shrimp) (not (v11_4)) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_anxiety_shrimp)) (fears_anxiety_lubricity)))
 (:action succumb__prostatitis__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_prostatitis_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_prostatitis_satisfaction)) (craves_prostatitis_chocolate)))
 (:action succumb__prostatitis__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_prostatitis_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_prostatitis_broccoli) (not (fears_prostatitis_satisfaction))))
 (:action succumb__prostatitis__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_prostatitis_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_prostatitis_tuna) (not (fears_prostatitis_satisfaction))))
 (:action succumb__prostatitis__satisfaction__turkey
  :parameters ()
  :precondition (and (fears_prostatitis_satisfaction) (craves_satisfaction_turkey) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_prostatitis_satisfaction)) (craves_prostatitis_turkey)))
 (:action succumb__prostatitis__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_prostatitis_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_prostatitis_satisfaction)) (craves_prostatitis_shrimp)))
 (:action succumb__prostatitis__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_prostatitis_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_prostatitis_satisfaction)) (craves_prostatitis_sweetroll)))
 (:action overcome__anxiety__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_anxiety_shrimp) (not (v8_4)) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anxiety_shrimp)) (fears_anxiety_intoxication)))
 (:action succumb__jealousy__lubricity__scallop
  :parameters ()
  :precondition (and (fears_jealousy_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_jealousy_lubricity)) (craves_jealousy_scallop)))
 (:action overcome__anxiety__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_anxiety_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anxiety_turkey)) (fears_anxiety_intoxication)))
 (:action succumb__jealousy__lubricity__chocolate
  :parameters ()
  :precondition (and (fears_jealousy_lubricity) (craves_lubricity_chocolate) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_jealousy_chocolate) (not (fears_jealousy_lubricity))))
 (:action succumb__jealousy__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_jealousy_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_jealousy_sweetroll) (not (fears_jealousy_lubricity))))
 (:action succumb__jealousy__lubricity__cherry
  :parameters ()
  :precondition (and (fears_jealousy_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_jealousy_cherry) (not (fears_jealousy_lubricity))))
 (:action succumb__jealousy__lubricity__shrimp
  :parameters ()
  :precondition (and (fears_jealousy_lubricity) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_jealousy_lubricity)) (craves_jealousy_shrimp)))
 (:action feast__lubricity__chocolate__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (not (v4_4)) (not (of)))
  :effect (and (when (z_3_0) (v4_0)) (when (not (z_3_0)) (not (v4_0))) (when (z_3_1) (v4_1)) (when (not (z_3_1)) (not (v4_1))) (when (z_3_2) (v4_2)) (when (not (z_3_2)) (not (v4_2))) (when (z_3_3) (v4_3)) (when (not (z_3_3)) (not (v4_3))) (when (z_3_4) (v4_4)) (when (not (z_3_4)) (not (v4_4))) (when (and (v4_4) (not (z_3_4))) (of)) (not (craves_lubricity_chocolate)) (craves_lubricity_shrimp)))
 (:action succumb__jealousy__excitement__cherry
  :parameters ()
  :precondition (and (craves_excitement_cherry) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_cherry)))
 (:action succumb__jealousy__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_shrimp)))
 (:action succumb__depression__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_depression_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_depression_sweetroll) (not (fears_depression_excitement))))
 (:action succumb__depression__excitement__cherry
  :parameters ()
  :precondition (and (craves_excitement_cherry) (fears_depression_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_depression_cherry) (not (fears_depression_excitement))))
 (:action overcome__boils__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_boils_shrimp) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_boils_lubricity) (not (craves_boils_shrimp))))
 (:action overcome__boils__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_boils_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_boils_lubricity) (not (craves_boils_chocolate))))
 (:action succumb__depression__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_depression_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_depression_chocolate) (not (fears_depression_satisfaction))))
 (:action succumb__depression__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_depression_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_depression_broccoli) (not (fears_depression_satisfaction))))
 (:action succumb__depression__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_depression_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_depression_tuna) (not (fears_depression_satisfaction))))
 (:action succumb__depression__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_depression_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_depression_turkey) (not (fears_depression_satisfaction))))
 (:action succumb__depression__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_depression_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_depression_shrimp) (not (fears_depression_satisfaction))))
 (:action succumb__depression__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_depression_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_depression_sweetroll) (not (fears_depression_satisfaction))))
 (:action succumb__depression__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_depression_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_depression_cherry) (not (fears_depression_satisfaction))))
 (:action overcome__boils__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_shrimp) (craves_boils_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_boils_intoxication) (not (craves_boils_shrimp))))
 (:action overcome__boils__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_boils_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_boils_intoxication) (not (craves_boils_turkey))))
 (:action overcome__anxiety__excitement__shrimp
  :parameters ()
  :precondition (and (craves_anxiety_shrimp) (craves_excitement_shrimp) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_anxiety_shrimp)) (fears_anxiety_excitement)))
 (:action overcome__anxiety__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_anxiety_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anxiety_excitement) (not (craves_anxiety_turkey))))
 (:action succumb__prostatitis__intoxication__shrimp
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_prostatitis_shrimp) (not (fears_prostatitis_intoxication))))
 (:action feast__lubricity__cherry__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_cherry) (not (v7_4)) (not (of)))
  :effect (and (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of)) (not (craves_lubricity_cherry)) (craves_lubricity_shrimp)))
 (:action feast__lubricity__cherry__scallop
  :parameters ()
  :precondition (and (craves_lubricity_cherry) (not (v7_4)) (not (of)))
  :effect (and (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of)) (not (craves_lubricity_cherry)) (craves_lubricity_scallop)))
 (:action overcome__anxiety__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_anxiety_shrimp) (craves_satisfaction_shrimp) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_anxiety_shrimp)) (fears_anxiety_satisfaction)))
 (:action overcome__anxiety__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_anxiety_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_anxiety_satisfaction) (not (craves_anxiety_turkey))))
 (:action overcome__abrasion__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_abrasion_shrimp) (not (v11_4)) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_abrasion_shrimp)) (fears_abrasion_lubricity)))
 (:action overcome__abrasion__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_abrasion_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_abrasion_chocolate)) (fears_abrasion_lubricity)))
 (:action feast__excitement__sweetroll__shrimp
  :parameters ()
  :precondition (and (craves_excitement_sweetroll) (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (not (craves_excitement_sweetroll)) (craves_excitement_shrimp)))
 (:action feast__excitement__sweetroll__scallop
  :parameters ()
  :precondition (and (craves_excitement_sweetroll) (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (craves_excitement_scallop) (not (craves_excitement_sweetroll))))
 (:action overcome__abrasion__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_abrasion_shrimp) (not (v8_4)) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_abrasion_intoxication) (not (craves_abrasion_shrimp))))
 (:action succumb__dread__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_dread_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_dread_lubricity)) (craves_dread_sweetroll)))
 (:action succumb__dread__lubricity__cherry
  :parameters ()
  :precondition (and (fears_dread_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_dread_lubricity)) (craves_dread_cherry)))
 (:action succumb__dread__lubricity__shrimp
  :parameters ()
  :precondition (and (fears_dread_lubricity) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_dread_shrimp) (not (fears_dread_lubricity))))
 (:action succumb__dread__lubricity__scallop
  :parameters ()
  :precondition (and (fears_dread_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_dread_scallop) (not (fears_dread_lubricity))))
 (:action overcome__abrasion__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_broccoli) (craves_abrasion_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_abrasion_broccoli)) (fears_abrasion_intoxication)))
 (:action succumb__dread__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_dread_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_dread_lubricity)) (craves_dread_chocolate)))
 (:action overcome__boils__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_boils_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_boils_excitement) (not (craves_boils_turkey))))
 (:action overcome__boils__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v2_4)) (craves_boils_shrimp) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_boils_excitement) (not (craves_boils_shrimp))))
 (:action feast__excitement__cherry__scallop
  :parameters ()
  :precondition (and (craves_excitement_cherry) (not (v7_4)) (not (of)))
  :effect (and (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of)) (craves_excitement_scallop) (not (craves_excitement_cherry))))
 (:action feast__excitement__cherry__shrimp
  :parameters ()
  :precondition (and (craves_excitement_cherry) (not (v7_4)) (not (of)))
  :effect (and (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of)) (not (craves_excitement_cherry)) (craves_excitement_shrimp)))
 (:action overcome__boils__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_boils_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_boils_turkey)) (fears_boils_satisfaction)))
 (:action overcome__boils__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v1_4)) (craves_boils_shrimp) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_boils_satisfaction) (not (craves_boils_shrimp))))
 (:action succumb__jealousy__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_broccoli)))
 (:action succumb__jealousy__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_turkey)))
 (:action succumb__jealousy__excitement__chocolate
  :parameters ()
  :precondition (and (fears_jealousy_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_jealousy_chocolate) (not (fears_jealousy_excitement))))
 (:action succumb__jealousy__excitement__sweetroll
  :parameters ()
  :precondition (and (craves_excitement_sweetroll) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_sweetroll)))
 (:action succumb__jealousy__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_tuna)))
 (:action succumb__jealousy__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_jealousy_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_jealousy_satisfaction)) (craves_jealousy_shrimp)))
 (:action succumb__jealousy__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_satisfaction_sweetroll) (fears_jealousy_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_jealousy_satisfaction)) (craves_jealousy_sweetroll)))
 (:action succumb__jealousy__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_jealousy_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_jealousy_satisfaction)) (craves_jealousy_cherry)))
 (:action succumb__jealousy__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_jealousy_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_jealousy_chocolate) (not (fears_jealousy_satisfaction))))
 (:action succumb__jealousy__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_satisfaction_broccoli) (fears_jealousy_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_jealousy_satisfaction)) (craves_jealousy_broccoli)))
 (:action succumb__jealousy__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_jealousy_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_jealousy_tuna) (not (fears_jealousy_satisfaction))))
 (:action succumb__jealousy__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_jealousy_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_jealousy_satisfaction)) (craves_jealousy_turkey)))
 (:action overcome__angina__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_shrimp) (craves_angina_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_angina_intoxication) (not (craves_angina_shrimp))))
 (:action overcome__abrasion__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v2_4)) (craves_abrasion_broccoli) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_abrasion_broccoli)) (fears_abrasion_excitement)))
 (:action overcome__abrasion__excitement__shrimp
  :parameters ()
  :precondition (and (craves_abrasion_shrimp) (craves_excitement_shrimp) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_abrasion_excitement) (not (craves_abrasion_shrimp))))
 (:action overcome__abrasion__satisfaction__broccoli
  :parameters ()
  :precondition (and (not (v1_4)) (craves_abrasion_broccoli) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_abrasion_broccoli)) (fears_abrasion_satisfaction)))
 (:action overcome__abrasion__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_abrasion_shrimp) (craves_satisfaction_shrimp) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_abrasion_shrimp)) (fears_abrasion_satisfaction)))
 (:action overcome__laceration__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_laceration_shrimp) (not (v8_4)) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_shrimp))))
 (:action succumb__sciatica__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_sciatica_shrimp) (not (fears_sciatica_intoxication))))
 (:action succumb__loneliness__lubricity__chocolate
  :parameters ()
  :precondition (and (fears_loneliness_lubricity) (craves_lubricity_chocolate) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_loneliness_lubricity)) (craves_loneliness_chocolate)))
 (:action succumb__grief__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_grief_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_grief_excitement)) (craves_grief_sweetroll)))
 (:action succumb__grief__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_grief_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_grief_excitement)) (craves_grief_tuna)))
 (:action succumb__grief__excitement__cherry
  :parameters ()
  :precondition (and (fears_grief_excitement) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_grief_cherry) (not (fears_grief_excitement))))
 (:action succumb__grief__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_grief_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_grief_excitement)) (craves_grief_shrimp)))
 (:action succumb__grief__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_grief_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_grief_broccoli) (not (fears_grief_excitement))))
 (:action succumb__grief__excitement__turkey
  :parameters ()
  :precondition (and (fears_grief_excitement) (craves_excitement_turkey) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_grief_turkey) (not (fears_grief_excitement))))
 (:action succumb__grief__excitement__chocolate
  :parameters ()
  :precondition (and (fears_grief_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_grief_excitement)) (craves_grief_chocolate)))
 (:action succumb__grief__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_grief_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief_sweetroll) (not (fears_grief_satisfaction))))
 (:action succumb__grief__satisfaction__cherry
  :parameters ()
  :precondition (and (fears_grief_satisfaction) (craves_satisfaction_cherry) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief_cherry) (not (fears_grief_satisfaction))))
 (:action succumb__depression__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (fears_depression_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_depression_intoxication)) (craves_depression_shrimp)))
 (:action succumb__sciatica__excitement__cherry
  :parameters ()
  :precondition (and (craves_excitement_cherry) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_sciatica_excitement)) (craves_sciatica_cherry)))
 (:action succumb__sciatica__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_sciatica_shrimp) (not (fears_sciatica_excitement))))
 (:action succumb__sciatica__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_sciatica_excitement)) (craves_sciatica_broccoli)))
 (:action succumb__sciatica__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_sciatica_excitement)) (craves_sciatica_turkey)))
 (:action succumb__sciatica__excitement__chocolate
  :parameters ()
  :precondition (and (fears_sciatica_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_sciatica_excitement)) (craves_sciatica_chocolate)))
 (:action succumb__sciatica__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_sciatica_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_sciatica_sweetroll) (not (fears_sciatica_excitement))))
 (:action succumb__sciatica__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_sciatica_excitement)) (craves_sciatica_tuna)))
 (:action succumb__sciatica__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_sciatica_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_sciatica_sweetroll) (not (fears_sciatica_satisfaction))))
 (:action succumb__sciatica__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_sciatica_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_sciatica_cherry) (not (fears_sciatica_satisfaction))))
 (:action succumb__grief__intoxication__shrimp
  :parameters ()
  :precondition (and (fears_grief_intoxication) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief_shrimp) (not (fears_grief_intoxication))))
 (:action succumb__angina__lubricity__chocolate
  :parameters ()
  :precondition (and (fears_angina_lubricity) (craves_lubricity_chocolate) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_angina_lubricity)) (craves_angina_chocolate)))
 (:action succumb__angina__lubricity__shrimp
  :parameters ()
  :precondition (and (fears_angina_lubricity) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_angina_lubricity)) (craves_angina_shrimp)))
 (:action succumb__angina__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_angina_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_angina_sweetroll) (not (fears_angina_lubricity))))
 (:action succumb__angina__lubricity__scallop
  :parameters ()
  :precondition (and (fears_angina_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_angina_scallop) (not (fears_angina_lubricity))))
 (:action succumb__angina__lubricity__cherry
  :parameters ()
  :precondition (and (fears_angina_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_angina_lubricity)) (craves_angina_cherry)))
 (:action overcome__sciatica__excitement__shrimp
  :parameters ()
  :precondition (and (craves_sciatica_shrimp) (craves_excitement_shrimp) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_shrimp))))
 (:action overcome__sciatica__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_sciatica_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_turkey))))
 (:action overcome__sciatica__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (craves_sciatica_shrimp) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_sciatica_shrimp)) (fears_sciatica_satisfaction)))
 (:action succumb__angina__excitement__cherry
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_angina_cherry) (not (fears_angina_excitement))))
 (:action overcome__sciatica__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_sciatica_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_sciatica_satisfaction) (not (craves_sciatica_turkey))))
 (:action succumb__angina__excitement__chocolate
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_angina_chocolate) (not (fears_angina_excitement))))
 (:action succumb__angina__excitement__broccoli
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_broccoli) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_angina_broccoli) (not (fears_angina_excitement))))
 (:action succumb__angina__excitement__tuna
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_tuna) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_angina_tuna) (not (fears_angina_excitement))))
 (:action succumb__angina__excitement__turkey
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_turkey) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_angina_turkey) (not (fears_angina_excitement))))
 (:action succumb__angina__excitement__shrimp
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_shrimp) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_angina_shrimp) (not (fears_angina_excitement))))
 (:action succumb__angina__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_angina_sweetroll) (not (fears_angina_excitement))))
 (:action succumb__angina__satisfaction__cherry
  :parameters ()
  :precondition (and (fears_angina_satisfaction) (craves_satisfaction_cherry) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_angina_satisfaction)) (craves_angina_cherry)))
 (:action succumb__angina__satisfaction__shrimp
  :parameters ()
  :precondition (and (fears_angina_satisfaction) (craves_satisfaction_shrimp) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_angina_satisfaction)) (craves_angina_shrimp)))
 (:action overcome__sciatica__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_sciatica_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_sciatica_chocolate)) (fears_sciatica_lubricity)))
 (:action overcome__sciatica__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_sciatica_shrimp) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_sciatica_shrimp)) (fears_sciatica_lubricity)))
 (:action succumb__angina__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_angina_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_angina_satisfaction)) (craves_angina_broccoli)))
 (:action succumb__angina__satisfaction__turkey
  :parameters ()
  :precondition (and (fears_angina_satisfaction) (craves_satisfaction_turkey) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_angina_turkey) (not (fears_angina_satisfaction))))
 (:action succumb__angina__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_angina_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_angina_satisfaction)) (craves_angina_chocolate)))
 (:action succumb__angina__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_angina_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_angina_sweetroll) (not (fears_angina_satisfaction))))
 (:action succumb__angina__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_angina_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_angina_satisfaction)) (craves_angina_tuna)))
 (:action succumb__laceration__lubricity__scallop
  :parameters ()
  :precondition (and (craves_lubricity_scallop) (fears_laceration_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_laceration_lubricity)) (craves_laceration_scallop)))
 (:action succumb__laceration__lubricity__cherry
  :parameters ()
  :precondition (and (fears_laceration_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_laceration_cherry) (not (fears_laceration_lubricity))))
 (:action overcome__sciatica__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_sciatica_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_sciatica_intoxication) (not (craves_sciatica_turkey))))
 (:action overcome__sciatica__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_sciatica_shrimp) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_sciatica_intoxication) (not (craves_sciatica_shrimp))))
 (:action overcome__grief__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v2_4)) (craves_grief_shrimp) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief_excitement) (not (craves_grief_shrimp))))
 (:action overcome__grief__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_grief_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief_excitement) (not (craves_grief_turkey))))
 (:action succumb__abrasion__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (fears_abrasion_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_abrasion_intoxication)) (craves_abrasion_shrimp)))
 (:action overcome__grief__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_grief_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief_turkey)) (fears_grief_satisfaction)))
 (:action overcome__grief__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v1_4)) (craves_grief_shrimp) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief_shrimp)) (fears_grief_satisfaction)))
 (:action feast__intoxication__shrimp__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (craves_intoxication_chocolate) (not (craves_intoxication_shrimp))))
 (:action succumb__abrasion__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_abrasion_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_abrasion_sweetroll) (not (fears_abrasion_excitement))))
 (:action succumb__abrasion__excitement__cherry
  :parameters ()
  :precondition (and (fears_abrasion_excitement) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_abrasion_excitement)) (craves_abrasion_cherry)))
 (:action feast__intoxication__shrimp__sweetroll
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (craves_intoxication_sweetroll) (not (craves_intoxication_shrimp))))
 (:action feast__intoxication__shrimp__cherry
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (not (v6_4)) (not (of)))
  :effect (and (when (z_7_0) (v6_0)) (when (not (z_7_0)) (not (v6_0))) (when (z_7_1) (v6_1)) (when (not (z_7_1)) (not (v6_1))) (when (z_7_2) (v6_2)) (when (not (z_7_2)) (not (v6_2))) (when (z_7_3) (v6_3)) (when (not (z_7_3)) (not (v6_3))) (when (z_7_4) (v6_4)) (when (not (z_7_4)) (not (v6_4))) (when (and (v6_4) (not (z_7_4))) (of)) (not (craves_intoxication_shrimp)) (craves_intoxication_cherry)))
 (:action overcome__grief__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_shrimp) (craves_grief_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief_lubricity) (not (craves_grief_shrimp))))
 (:action overcome__grief__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_grief_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_grief_chocolate)) (fears_grief_lubricity)))
 (:action succumb__abrasion__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_abrasion_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_abrasion_satisfaction)) (craves_abrasion_broccoli)))
 (:action succumb__abrasion__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_abrasion_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_abrasion_satisfaction)) (craves_abrasion_turkey)))
 (:action succumb__abrasion__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_abrasion_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_abrasion_satisfaction)) (craves_abrasion_chocolate)))
 (:action succumb__abrasion__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_abrasion_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_abrasion_satisfaction)) (craves_abrasion_sweetroll)))
 (:action succumb__abrasion__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_abrasion_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_abrasion_satisfaction)) (craves_abrasion_tuna)))
 (:action succumb__abrasion__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_abrasion_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_abrasion_satisfaction)) (craves_abrasion_cherry)))
 (:action succumb__abrasion__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_abrasion_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_abrasion_satisfaction)) (craves_abrasion_shrimp)))
 (:action overcome__grief__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_shrimp) (craves_grief_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_grief_intoxication) (not (craves_grief_shrimp))))
 (:action overcome__grief__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_grief_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_grief_turkey)) (fears_grief_intoxication)))
 (:action succumb__grief2__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_grief2_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief2_lubricity)) (craves_grief2_chocolate)))
 (:action overcome__depression__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (craves_depression_shrimp) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_depression_excitement) (not (craves_depression_shrimp))))
 (:action overcome__depression__excitement__broccoli
  :parameters ()
  :precondition (and (craves_depression_broccoli) (craves_excitement_broccoli) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_depression_excitement) (not (craves_depression_broccoli))))
 (:action overcome__depression__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_depression_broccoli) (not (v1_4)) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_depression_broccoli)) (fears_depression_satisfaction)))
 (:action overcome__depression__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (craves_depression_shrimp) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_depression_satisfaction) (not (craves_depression_shrimp))))
 (:action overcome__depression__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_depression_shrimp) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_depression_lubricity) (not (craves_depression_shrimp))))
 (:action overcome__depression__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_depression_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_depression_chocolate)) (fears_depression_lubricity)))
 (:action feast__satisfaction__sweetroll__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_sweetroll) (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (craves_satisfaction_shrimp) (not (craves_satisfaction_sweetroll))))
 (:action feast__satisfaction__sweetroll__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_sweetroll) (not (v9_4)) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (not (craves_satisfaction_sweetroll)) (craves_satisfaction_scallop)))
 (:action overcome__depression__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_depression_shrimp) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_depression_shrimp)) (fears_depression_intoxication)))
 (:action succumb__loneliness__lubricity__shrimp
  :parameters ()
  :precondition (and (fears_loneliness_lubricity) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_loneliness_lubricity)) (craves_loneliness_shrimp)))
 (:action succumb__loneliness__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_loneliness_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_loneliness_lubricity)) (craves_loneliness_sweetroll)))
 (:action succumb__loneliness__lubricity__scallop
  :parameters ()
  :precondition (and (fears_loneliness_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_loneliness_lubricity)) (craves_loneliness_scallop)))
 (:action succumb__loneliness__lubricity__cherry
  :parameters ()
  :precondition (and (fears_loneliness_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_loneliness_lubricity)) (craves_loneliness_cherry)))
 (:action overcome__depression__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_depression_broccoli) (not (v8_4)) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_depression_broccoli)) (fears_depression_intoxication)))
 (:action succumb__laceration__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_laceration_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_laceration_chocolate) (not (fears_laceration_lubricity))))
 (:action succumb__laceration__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (fears_laceration_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_laceration_shrimp) (not (fears_laceration_lubricity))))
 (:action succumb__laceration__lubricity__sweetroll
  :parameters ()
  :precondition (and (craves_lubricity_sweetroll) (fears_laceration_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_laceration_sweetroll) (not (fears_laceration_lubricity))))
 (:action overcome__prostatitis__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (craves_prostatitis_broccoli) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_prostatitis_broccoli)) (fears_prostatitis_excitement)))
 (:action overcome__prostatitis__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v2_4)) (craves_prostatitis_shrimp) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_prostatitis_shrimp)) (fears_prostatitis_excitement)))
 (:action feast__satisfaction__cherry__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v7_4)) (not (of)))
  :effect (and (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of)) (not (craves_satisfaction_cherry)) (craves_satisfaction_scallop)))
 (:action feast__satisfaction__cherry__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v7_4)) (not (of)))
  :effect (and (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of)) (not (craves_satisfaction_cherry)) (craves_satisfaction_shrimp)))
 (:action overcome__prostatitis__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_prostatitis_broccoli) (not (v1_4)) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_prostatitis_satisfaction) (not (craves_prostatitis_broccoli))))
 (:action overcome__prostatitis__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v1_4)) (craves_prostatitis_shrimp) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_prostatitis_satisfaction) (not (craves_prostatitis_shrimp))))
 (:action succumb__laceration__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_laceration_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_laceration_shrimp) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_laceration_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_laceration_excitement)) (craves_laceration_sweetroll)))
 (:action succumb__laceration__excitement__cherry
  :parameters ()
  :precondition (and (craves_excitement_cherry) (fears_laceration_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_laceration_cherry) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__chocolate
  :parameters ()
  :precondition (and (fears_laceration_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_laceration_chocolate) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_laceration_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_laceration_excitement)) (craves_laceration_broccoli)))
 (:action succumb__laceration__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_laceration_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_laceration_tuna) (not (fears_laceration_excitement))))
 (:action succumb__laceration__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (fears_laceration_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_laceration_excitement)) (craves_laceration_turkey)))
 (:action overcome__prostatitis__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_prostatitis_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_prostatitis_chocolate)) (fears_prostatitis_lubricity)))
 (:action overcome__prostatitis__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_shrimp) (craves_prostatitis_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_prostatitis_shrimp)) (fears_prostatitis_lubricity)))
 (:action succumb__laceration__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_satisfaction_sweetroll) (fears_laceration_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_laceration_sweetroll) (not (fears_laceration_satisfaction))))
 (:action succumb__laceration__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_laceration_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_laceration_tuna) (not (fears_laceration_satisfaction))))
 (:action succumb__laceration__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_laceration_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_laceration_cherry) (not (fears_laceration_satisfaction))))
 (:action succumb__laceration__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_laceration_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_laceration_shrimp) (not (fears_laceration_satisfaction))))
 (:action succumb__laceration__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_laceration_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_laceration_satisfaction)) (craves_laceration_broccoli)))
 (:action succumb__laceration__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_laceration_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_laceration_turkey) (not (fears_laceration_satisfaction))))
 (:action succumb__laceration__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_laceration_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_laceration_chocolate) (not (fears_laceration_satisfaction))))
 (:action overcome__prostatitis__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_prostatitis_broccoli) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_prostatitis_broccoli)) (fears_prostatitis_intoxication)))
 (:action overcome__prostatitis__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_shrimp) (craves_prostatitis_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_prostatitis_shrimp)) (fears_prostatitis_intoxication)))
 (:action succumb__grief2__lubricity__shrimp
  :parameters ()
  :precondition (and (fears_grief2_lubricity) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_grief2_shrimp) (not (fears_grief2_lubricity))))
 (:action succumb__grief2__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_grief2_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief2_lubricity)) (craves_grief2_sweetroll)))
 (:action succumb__grief2__lubricity__scallop
  :parameters ()
  :precondition (and (fears_grief2_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief2_lubricity)) (craves_grief2_scallop)))
 (:action succumb__grief2__lubricity__cherry
  :parameters ()
  :precondition (and (fears_grief2_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief2_lubricity)) (craves_grief2_cherry)))
 (:action overcome__jealousy__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_shrimp) (craves_jealousy_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_jealousy_intoxication) (not (craves_jealousy_shrimp))))
 (:action overcome__anger__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_cherry) (craves_anger_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_anger_cherry)) (fears_anger_excitement)))
 (:action overcome__anger__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_anger_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_anger_cherry)) (fears_anger_satisfaction)))
 (:action succumb__hangover__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_hangover_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_hangover_chocolate) (not (fears_hangover_lubricity))))
 (:action succumb__hangover__lubricity__sweetroll
  :parameters ()
  :precondition (and (craves_lubricity_sweetroll) (fears_hangover_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_hangover_sweetroll) (not (fears_hangover_lubricity))))
 (:action succumb__hangover__lubricity__cherry
  :parameters ()
  :precondition (and (craves_lubricity_cherry) (fears_hangover_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_hangover_cherry) (not (fears_hangover_lubricity))))
 (:action succumb__hangover__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (fears_hangover_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_hangover_shrimp) (not (fears_hangover_lubricity))))
 (:action succumb__hangover__lubricity__scallop
  :parameters ()
  :precondition (and (craves_lubricity_scallop) (fears_hangover_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_hangover_scallop) (not (fears_hangover_lubricity))))
 (:action overcome__anger__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_anger_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anger_lubricity) (not (craves_anger_cherry))))
 (:action succumb__boils__intoxication__shrimp
  :parameters ()
  :precondition (and (fears_boils_intoxication) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_shrimp)))
 (:action succumb__boils__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_boils_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_boils_tuna) (not (fears_boils_excitement))))
 (:action succumb__boils__excitement__turkey
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_turkey) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_boils_excitement)) (craves_boils_turkey)))
 (:action succumb__boils__excitement__shrimp
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_shrimp) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_boils_excitement)) (craves_boils_shrimp)))
 (:action succumb__boils__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_boils_excitement)) (craves_boils_sweetroll)))
 (:action succumb__boils__excitement__cherry
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_boils_cherry) (not (fears_boils_excitement))))
 (:action succumb__boils__excitement__chocolate
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_boils_excitement)) (craves_boils_chocolate)))
 (:action succumb__boils__excitement__broccoli
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_broccoli) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_boils_excitement)) (craves_boils_broccoli)))
 (:action succumb__boils__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_satisfaction_sweetroll) (fears_boils_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_boils_satisfaction)) (craves_boils_sweetroll)))
 (:action succumb__boils__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_boils_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_boils_cherry) (not (fears_boils_satisfaction))))
 (:action succumb__anxiety__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (fears_anxiety_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_anxiety_shrimp) (not (fears_anxiety_intoxication))))
 (:action succumb__anxiety__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_anxiety_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anxiety_shrimp) (not (fears_anxiety_excitement))))
 (:action succumb__anxiety__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anxiety_sweetroll) (not (fears_anxiety_excitement))))
 (:action succumb__anxiety__excitement__cherry
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anxiety_cherry) (not (fears_anxiety_excitement))))
 (:action succumb__anxiety__excitement__chocolate
  :parameters ()
  :precondition (and (fears_anxiety_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_anxiety_excitement)) (craves_anxiety_chocolate)))
 (:action succumb__anxiety__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_anxiety_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_anxiety_excitement)) (craves_anxiety_broccoli)))
 (:action succumb__anxiety__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_anxiety_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_anxiety_excitement)) (craves_anxiety_tuna)))
 (:action succumb__anxiety__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (fears_anxiety_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anxiety_turkey) (not (fears_anxiety_excitement))))
 (:action succumb__anxiety__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_anxiety_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anxiety_sweetroll) (not (fears_anxiety_satisfaction))))
 (:action succumb__anxiety__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_anxiety_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_anxiety_satisfaction)) (craves_anxiety_cherry)))
 (:action overcome__anxiety__lubricity__cherry
  :parameters ()
  :precondition (and (craves_anxiety_cherry) (not (v11_4)) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_anxiety_cherry)) (fears_anxiety_lubricity)))
 (:action succumb__prostatitis__satisfaction__scallop
  :parameters ()
  :precondition (and (fears_prostatitis_satisfaction) (craves_satisfaction_scallop) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_prostatitis_satisfaction)) (craves_prostatitis_scallop)))
 (:action overcome__anxiety__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_anxiety_broccoli) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_anxiety_broccoli)) (fears_anxiety_lubricity)))
 (:action overcome__anxiety__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_turkey) (craves_anxiety_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_anxiety_turkey)) (fears_anxiety_lubricity)))
 (:action overcome__anxiety__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_sweetroll) (craves_anxiety_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_anxiety_sweetroll)) (fears_anxiety_lubricity)))
 (:action feast__lubricity__broccoli__tuna
  :parameters ()
  :precondition (and (craves_lubricity_broccoli) (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (not (craves_lubricity_broccoli)) (craves_lubricity_tuna)))
 (:action feast__lubricity__broccoli__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_broccoli) (not (v3_4)) (not (of)))
  :effect (and (when (z_2_0) (v3_0)) (when (not (z_2_0)) (not (v3_0))) (when (z_2_1) (v3_1)) (when (not (z_2_1)) (not (v3_1))) (when (z_2_2) (v3_2)) (when (not (z_2_2)) (not (v3_2))) (when (z_2_3) (v3_3)) (when (not (z_2_3)) (not (v3_3))) (when (z_2_4) (v3_4)) (when (not (z_2_4)) (not (v3_4))) (when (and (v3_4) (not (z_2_4))) (of)) (not (craves_lubricity_broccoli)) (craves_lubricity_chocolate)))
 (:action overcome__anxiety__intoxication__cherry
  :parameters ()
  :precondition (and (craves_anxiety_cherry) (not (v8_4)) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anxiety_cherry)) (fears_anxiety_intoxication)))
 (:action overcome__anxiety__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_sweetroll) (craves_anxiety_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anxiety_sweetroll)) (fears_anxiety_intoxication)))
 (:action succumb__jealousy__lubricity__broccoli
  :parameters ()
  :precondition (and (fears_jealousy_lubricity) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_jealousy_broccoli) (not (fears_jealousy_lubricity))))
 (:action succumb__jealousy__lubricity__turkey
  :parameters ()
  :precondition (and (fears_jealousy_lubricity) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_jealousy_lubricity)) (craves_jealousy_turkey)))
 (:action succumb__jealousy__intoxication__cherry
  :parameters ()
  :precondition (and (craves_intoxication_cherry) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_jealousy_intoxication)) (craves_jealousy_cherry)))
 (:action succumb__jealousy__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_jealousy_chocolate) (not (fears_jealousy_intoxication))))
 (:action succumb__jealousy__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_jealousy_intoxication)) (craves_jealousy_broccoli)))
 (:action succumb__jealousy__intoxication__tuna
  :parameters ()
  :precondition (and (fears_jealousy_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_jealousy_intoxication)) (craves_jealousy_tuna)))
 (:action succumb__jealousy__intoxication__turkey
  :parameters ()
  :precondition (and (craves_intoxication_turkey) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_jealousy_intoxication)) (craves_jealousy_turkey)))
 (:action succumb__jealousy__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_jealousy_intoxication)) (craves_jealousy_shrimp)))
 (:action succumb__jealousy__intoxication__sweetroll
  :parameters ()
  :precondition (and (craves_intoxication_sweetroll) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_jealousy_intoxication)) (craves_jealousy_sweetroll)))
 (:action feast__lubricity__turkey__chocolate
  :parameters ()
  :precondition (and (not (v0_4)) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of)) (craves_lubricity_chocolate) (not (craves_lubricity_turkey))))
 (:action feast__lubricity__turkey__tuna
  :parameters ()
  :precondition (and (not (v0_4)) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_23_0) (v0_0)) (when (not (z_23_0)) (not (v0_0))) (when (z_23_1) (v0_1)) (when (not (z_23_1)) (not (v0_1))) (when (z_23_2) (v0_2)) (when (not (z_23_2)) (not (v0_2))) (when (z_23_3) (v0_3)) (when (not (z_23_3)) (not (v0_3))) (when (z_23_4) (v0_4)) (when (not (z_23_4)) (not (v0_4))) (when (and (v0_4) (not (z_23_4))) (of)) (craves_lubricity_tuna) (not (craves_lubricity_turkey))))
 (:action succumb__jealousy__excitement__scallop
  :parameters ()
  :precondition (and (craves_excitement_scallop) (fears_jealousy_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_jealousy_excitement)) (craves_jealousy_scallop)))
 (:action succumb__depression__excitement__scallop
  :parameters ()
  :precondition (and (craves_excitement_scallop) (fears_depression_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_depression_excitement)) (craves_depression_scallop)))
 (:action overcome__boils__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_boils_sweetroll) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_boils_lubricity) (not (craves_boils_sweetroll))))
 (:action overcome__boils__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_boils_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_boils_cherry)) (fears_boils_lubricity)))
 (:action overcome__boils__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_boils_broccoli) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_boils_broccoli)) (fears_boils_lubricity)))
 (:action overcome__boils__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_turkey) (craves_boils_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_boils_turkey)) (fears_boils_lubricity)))
 (:action succumb__depression__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_depression_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_depression_satisfaction)) (craves_depression_scallop)))
 (:action overcome__boils__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_boils_sweetroll) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_boils_intoxication) (not (craves_boils_sweetroll))))
 (:action overcome__boils__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_boils_cherry) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_boils_intoxication) (not (craves_boils_cherry))))
 (:action succumb__prostatitis__lubricity__turkey
  :parameters ()
  :precondition (and (fears_prostatitis_lubricity) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_prostatitis_turkey) (not (fears_prostatitis_lubricity))))
 (:action succumb__prostatitis__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_prostatitis_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_prostatitis_lubricity)) (craves_prostatitis_chocolate)))
 (:action succumb__prostatitis__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_prostatitis_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_prostatitis_sweetroll) (not (fears_prostatitis_lubricity))))
 (:action succumb__prostatitis__lubricity__cherry
  :parameters ()
  :precondition (and (fears_prostatitis_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_prostatitis_cherry) (not (fears_prostatitis_lubricity))))
 (:action succumb__prostatitis__lubricity__shrimp
  :parameters ()
  :precondition (and (fears_prostatitis_lubricity) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_prostatitis_shrimp) (not (fears_prostatitis_lubricity))))
 (:action succumb__prostatitis__lubricity__scallop
  :parameters ()
  :precondition (and (fears_prostatitis_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_prostatitis_scallop) (not (fears_prostatitis_lubricity))))
 (:action succumb__prostatitis__lubricity__broccoli
  :parameters ()
  :precondition (and (fears_prostatitis_lubricity) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_prostatitis_broccoli) (not (fears_prostatitis_lubricity))))
 (:action overcome__anxiety__excitement__sweetroll
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_sweetroll) (craves_anxiety_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anxiety_excitement) (not (craves_anxiety_sweetroll))))
 (:action overcome__anxiety__excitement__cherry
  :parameters ()
  :precondition (and (craves_anxiety_cherry) (not (v2_4)) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anxiety_excitement) (not (craves_anxiety_cherry))))
 (:action succumb__prostatitis__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_prostatitis_sweetroll) (not (fears_prostatitis_intoxication))))
 (:action succumb__prostatitis__intoxication__cherry
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_prostatitis_intoxication)) (craves_prostatitis_cherry)))
 (:action overcome__anxiety__satisfaction__sweetroll
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_sweetroll) (craves_anxiety_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_anxiety_satisfaction) (not (craves_anxiety_sweetroll))))
 (:action overcome__anxiety__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_anxiety_cherry) (craves_satisfaction_cherry) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_anxiety_satisfaction) (not (craves_anxiety_cherry))))
 (:action succumb__prostatitis__excitement__scallop
  :parameters ()
  :precondition (and (fears_prostatitis_excitement) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_prostatitis_scallop) (not (fears_prostatitis_excitement))))
 (:action succumb__anger__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_anger_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_anger_excitement)) (craves_anger_broccoli)))
 (:action succumb__anger__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (fears_anger_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anger_turkey) (not (fears_anger_excitement))))
 (:action succumb__anger__excitement__chocolate
  :parameters ()
  :precondition (and (fears_anger_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anger_chocolate) (not (fears_anger_excitement))))
 (:action overcome__abrasion__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_turkey) (craves_abrasion_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_abrasion_turkey)) (fears_abrasion_lubricity)))
 (:action overcome__abrasion__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_sweetroll) (craves_abrasion_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_abrasion_sweetroll)) (fears_abrasion_lubricity)))
 (:action overcome__abrasion__lubricity__cherry
  :parameters ()
  :precondition (and (craves_abrasion_cherry) (not (v11_4)) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_abrasion_cherry)) (fears_abrasion_lubricity)))
 (:action succumb__anger__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_anger_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_anger_satisfaction)) (craves_anger_tuna)))
 (:action succumb__anger__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_anger_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anger_turkey) (not (fears_anger_satisfaction))))
 (:action succumb__anger__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_anger_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anger_shrimp) (not (fears_anger_satisfaction))))
 (:action succumb__anger__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_anger_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anger_sweetroll) (not (fears_anger_satisfaction))))
 (:action succumb__anger__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_anger_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anger_scallop) (not (fears_anger_satisfaction))))
 (:action succumb__anger__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_anger_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anger_cherry) (not (fears_anger_satisfaction))))
 (:action overcome__abrasion__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_abrasion_broccoli) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_abrasion_broccoli)) (fears_abrasion_lubricity)))
 (:action succumb__anger__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_anger_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anger_chocolate) (not (fears_anger_satisfaction))))
 (:action succumb__anger__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_anger_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anger_broccoli) (not (fears_anger_satisfaction))))
 (:action overcome__abrasion__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_sweetroll) (craves_abrasion_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_abrasion_intoxication) (not (craves_abrasion_sweetroll))))
 (:action overcome__abrasion__intoxication__cherry
  :parameters ()
  :precondition (and (craves_abrasion_cherry) (not (v8_4)) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_abrasion_intoxication) (not (craves_abrasion_cherry))))
 (:action feast__excitement__scallop__sweetroll
  :parameters ()
  :precondition (and (not (v10_4)) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (not (craves_excitement_scallop)) (craves_excitement_sweetroll)))
 (:action succumb__dread__lubricity__broccoli
  :parameters ()
  :precondition (and (fears_dread_lubricity) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_dread_lubricity)) (craves_dread_broccoli)))
 (:action succumb__dread__lubricity__turkey
  :parameters ()
  :precondition (and (fears_dread_lubricity) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_dread_turkey) (not (fears_dread_lubricity))))
 (:action feast__excitement__scallop__cherry
  :parameters ()
  :precondition (and (not (v10_4)) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (not (craves_excitement_scallop)) (craves_excitement_cherry)))
 (:action overcome__boils__excitement__sweetroll
  :parameters ()
  :precondition (and (craves_boils_sweetroll) (not (v2_4)) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_boils_excitement) (not (craves_boils_sweetroll))))
 (:action overcome__boils__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_cherry) (craves_boils_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_boils_excitement) (not (craves_boils_cherry))))
 (:action overcome__boils__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_boils_sweetroll) (not (v1_4)) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_boils_satisfaction) (not (craves_boils_sweetroll))))
 (:action overcome__boils__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_boils_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_boils_cherry)) (fears_boils_satisfaction)))
 (:action overcome__angina__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_angina_broccoli) (not (v11_4)) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_angina_lubricity) (not (craves_angina_broccoli))))
 (:action overcome__angina__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_angina_chocolate) (craves_lubricity_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_angina_lubricity) (not (craves_angina_chocolate))))
 (:action overcome__angina__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_angina_turkey) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_angina_turkey)) (fears_angina_lubricity)))
 (:action overcome__angina__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_angina_sweetroll) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_angina_sweetroll)) (fears_angina_lubricity)))
 (:action overcome__angina__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_angina_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_angina_lubricity) (not (craves_angina_cherry))))
 (:action overcome__angina__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_scallop) (craves_angina_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_angina_lubricity) (not (craves_angina_scallop))))
 (:action succumb__jealousy__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_jealousy_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_jealousy_satisfaction)) (craves_jealousy_scallop)))
 (:action overcome__angina__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_angina_broccoli) (not (v8_4)) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_angina_intoxication) (not (craves_angina_broccoli))))
 (:action overcome__angina__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_angina_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_angina_intoxication) (not (craves_angina_chocolate))))
 (:action overcome__angina__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_angina_turkey) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_angina_turkey)) (fears_angina_intoxication)))
 (:action overcome__angina__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_angina_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_angina_tuna)) (fears_angina_intoxication)))
 (:action overcome__angina__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_angina_sweetroll) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_angina_sweetroll)) (fears_angina_intoxication)))
 (:action overcome__angina__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_cherry) (craves_angina_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_angina_intoxication) (not (craves_angina_cherry))))
 (:action succumb__anger__lubricity__cherry
  :parameters ()
  :precondition (and (fears_anger_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_anger_lubricity)) (craves_anger_cherry)))
 (:action succumb__anger__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (fears_anger_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_anger_lubricity)) (craves_anger_shrimp)))
 (:action succumb__anger__lubricity__scallop
  :parameters ()
  :precondition (and (fears_anger_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_anger_lubricity)) (craves_anger_scallop)))
 (:action succumb__anger__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_lubricity_broccoli) (fears_anger_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_anger_lubricity)) (craves_anger_broccoli)))
 (:action succumb__anger__lubricity__turkey
  :parameters ()
  :precondition (and (craves_lubricity_turkey) (fears_anger_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_anger_turkey) (not (fears_anger_lubricity))))
 (:action succumb__anger__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_anger_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_anger_lubricity)) (craves_anger_chocolate)))
 (:action succumb__anger__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_anger_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_anger_lubricity)) (craves_anger_sweetroll)))
 (:action overcome__abrasion__excitement__sweetroll
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_sweetroll) (craves_abrasion_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_abrasion_excitement) (not (craves_abrasion_sweetroll))))
 (:action overcome__abrasion__excitement__cherry
  :parameters ()
  :precondition (and (craves_abrasion_cherry) (not (v2_4)) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_abrasion_excitement) (not (craves_abrasion_cherry))))
 (:action overcome__abrasion__satisfaction__sweetroll
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_sweetroll) (craves_abrasion_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_abrasion_sweetroll)) (fears_abrasion_satisfaction)))
 (:action overcome__abrasion__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_abrasion_cherry) (craves_satisfaction_cherry) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_abrasion_cherry)) (fears_abrasion_satisfaction)))
 (:action succumb__anger__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_anger_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anger_sweetroll) (not (fears_anger_excitement))))
 (:action succumb__anger__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_anger_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_anger_excitement)) (craves_anger_tuna)))
 (:action succumb__anger__excitement__cherry
  :parameters ()
  :precondition (and (fears_anger_excitement) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anger_cherry) (not (fears_anger_excitement))))
 (:action succumb__anger__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_anger_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anger_shrimp) (not (fears_anger_excitement))))
 (:action succumb__anger__excitement__scallop
  :parameters ()
  :precondition (and (craves_excitement_scallop) (fears_anger_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_anger_excitement)) (craves_anger_scallop)))
 (:action overcome__laceration__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_laceration_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_laceration_cherry)) (fears_laceration_lubricity)))
 (:action overcome__laceration__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_laceration_scallop) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_laceration_scallop)) (fears_laceration_lubricity)))
 (:action overcome__laceration__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_broccoli) (craves_laceration_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_laceration_broccoli)) (fears_laceration_lubricity)))
 (:action overcome__laceration__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_laceration_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_laceration_chocolate)) (fears_laceration_lubricity)))
 (:action overcome__laceration__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_turkey) (craves_laceration_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_laceration_lubricity) (not (craves_laceration_turkey))))
 (:action overcome__laceration__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_laceration_sweetroll) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_laceration_lubricity) (not (craves_laceration_sweetroll))))
 (:action overcome__laceration__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_laceration_cherry) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_cherry))))
 (:action overcome__laceration__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_broccoli) (craves_laceration_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_laceration_broccoli)) (fears_laceration_intoxication)))
 (:action overcome__laceration__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_laceration_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_chocolate))))
 (:action overcome__laceration__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_laceration_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_turkey))))
 (:action overcome__laceration__intoxication__tuna
  :parameters ()
  :precondition (and (craves_laceration_tuna) (not (v8_4)) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_tuna))))
 (:action overcome__laceration__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_laceration_sweetroll) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_laceration_intoxication) (not (craves_laceration_sweetroll))))
 (:action succumb__sciatica__lubricity__scallop
  :parameters ()
  :precondition (and (fears_sciatica_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_sciatica_scallop) (not (fears_sciatica_lubricity))))
 (:action overcome__angina__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_cherry) (craves_angina_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_angina_excitement) (not (craves_angina_cherry))))
 (:action succumb__sciatica__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_lubricity_broccoli) (fears_sciatica_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_sciatica_lubricity)) (craves_sciatica_broccoli)))
 (:action overcome__angina__excitement__scallop
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_scallop) (craves_angina_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_angina_scallop)) (fears_angina_excitement)))
 (:action succumb__sciatica__lubricity__turkey
  :parameters ()
  :precondition (and (craves_lubricity_turkey) (fears_sciatica_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_sciatica_lubricity)) (craves_sciatica_turkey)))
 (:action succumb__sciatica__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_sciatica_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_sciatica_lubricity)) (craves_sciatica_chocolate)))
 (:action succumb__sciatica__lubricity__sweetroll
  :parameters ()
  :precondition (and (craves_lubricity_sweetroll) (fears_sciatica_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_sciatica_sweetroll) (not (fears_sciatica_lubricity))))
 (:action succumb__sciatica__lubricity__cherry
  :parameters ()
  :precondition (and (fears_sciatica_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_sciatica_lubricity)) (craves_sciatica_cherry)))
 (:action succumb__sciatica__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (fears_sciatica_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_sciatica_lubricity)) (craves_sciatica_shrimp)))
 (:action overcome__angina__excitement__broccoli
  :parameters ()
  :precondition (and (craves_angina_broccoli) (craves_excitement_broccoli) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_angina_excitement) (not (craves_angina_broccoli))))
 (:action overcome__angina__excitement__chocolate
  :parameters ()
  :precondition (and (craves_angina_chocolate) (not (v2_4)) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_angina_excitement) (not (craves_angina_chocolate))))
 (:action overcome__angina__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (craves_angina_turkey) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_angina_turkey)) (fears_angina_excitement)))
 (:action overcome__angina__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_angina_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_angina_tuna)) (fears_angina_excitement)))
 (:action overcome__angina__excitement__sweetroll
  :parameters ()
  :precondition (and (not (v2_4)) (craves_angina_sweetroll) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_angina_sweetroll)) (fears_angina_excitement)))
 (:action succumb__sciatica__intoxication__cherry
  :parameters ()
  :precondition (and (fears_sciatica_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_sciatica_cherry) (not (fears_sciatica_intoxication))))
 (:action overcome__angina__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_angina_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_angina_satisfaction) (not (craves_angina_cherry))))
 (:action overcome__angina__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (not (v1_4)) (craves_angina_scallop) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_angina_satisfaction) (not (craves_angina_scallop))))
 (:action succumb__sciatica__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_sciatica_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_sciatica_sweetroll) (not (fears_sciatica_intoxication))))
 (:action overcome__angina__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_angina_broccoli) (not (v1_4)) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_angina_satisfaction) (not (craves_angina_broccoli))))
 (:action overcome__angina__satisfaction__chocolate
  :parameters ()
  :precondition (and (craves_angina_chocolate) (not (v1_4)) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_angina_satisfaction) (not (craves_angina_chocolate))))
 (:action overcome__angina__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (craves_angina_turkey) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_angina_satisfaction) (not (craves_angina_turkey))))
 (:action overcome__angina__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_angina_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_angina_satisfaction) (not (craves_angina_tuna))))
 (:action overcome__angina__satisfaction__sweetroll
  :parameters ()
  :precondition (and (not (v1_4)) (craves_angina_sweetroll) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_angina_satisfaction) (not (craves_angina_sweetroll))))
 (:action succumb__loneliness__lubricity__broccoli
  :parameters ()
  :precondition (and (fears_loneliness_lubricity) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_loneliness_broccoli) (not (fears_loneliness_lubricity))))
 (:action overcome__grief2__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_sweetroll) (craves_grief2_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief2_lubricity) (not (craves_grief2_sweetroll))))
 (:action overcome__grief2__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_shrimp) (craves_grief2_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief2_lubricity) (not (craves_grief2_shrimp))))
 (:action overcome__grief2__lubricity__cherry
  :parameters ()
  :precondition (and (craves_grief2_cherry) (not (v11_4)) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief2_lubricity) (not (craves_grief2_cherry))))
 (:action overcome__grief2__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_grief2_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief2_lubricity) (not (craves_grief2_chocolate))))
 (:action overcome__grief2__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_sweetroll) (craves_grief2_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_grief2_sweetroll)) (fears_grief2_intoxication)))
 (:action overcome__grief2__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_shrimp) (craves_grief2_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_grief2_shrimp)) (fears_grief2_intoxication)))
 (:action overcome__grief2__intoxication__cherry
  :parameters ()
  :precondition (and (craves_grief2_cherry) (not (v8_4)) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_grief2_intoxication) (not (craves_grief2_cherry))))
 (:action overcome__grief2__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_grief2_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_grief2_intoxication) (not (craves_grief2_chocolate))))
 (:action overcome__laceration__excitement__sweetroll
  :parameters ()
  :precondition (and (craves_laceration_sweetroll) (not (v2_4)) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_laceration_excitement) (not (craves_laceration_sweetroll))))
 (:action overcome__laceration__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_laceration_cherry) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_laceration_cherry)) (fears_laceration_excitement)))
 (:action overcome__laceration__excitement__scallop
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_scallop) (craves_laceration_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_laceration_scallop)) (fears_laceration_excitement)))
 (:action overcome__laceration__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v2_4)) (craves_laceration_broccoli) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_laceration_broccoli)) (fears_laceration_excitement)))
 (:action overcome__laceration__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_laceration_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_laceration_excitement) (not (craves_laceration_chocolate))))
 (:action overcome__laceration__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_laceration_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_laceration_excitement) (not (craves_laceration_turkey))))
 (:action overcome__laceration__excitement__tuna
  :parameters ()
  :precondition (and (craves_laceration_tuna) (craves_excitement_tuna) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_laceration_excitement) (not (craves_laceration_tuna))))
 (:action overcome__laceration__satisfaction__tuna
  :parameters ()
  :precondition (and (craves_laceration_tuna) (not (v1_4)) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_laceration_tuna)) (fears_laceration_satisfaction)))
 (:action overcome__laceration__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_laceration_sweetroll) (not (v1_4)) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_laceration_satisfaction) (not (craves_laceration_sweetroll))))
 (:action overcome__laceration__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_laceration_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_laceration_cherry)) (fears_laceration_satisfaction)))
 (:action overcome__laceration__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (not (v1_4)) (craves_laceration_scallop) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_laceration_scallop)) (fears_laceration_satisfaction)))
 (:action overcome__laceration__satisfaction__broccoli
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_broccoli) (craves_laceration_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_laceration_broccoli)) (fears_laceration_satisfaction)))
 (:action overcome__laceration__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_laceration_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_laceration_chocolate)) (fears_laceration_satisfaction)))
 (:action overcome__laceration__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_laceration_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_laceration_satisfaction) (not (craves_laceration_turkey))))
 (:action succumb__grief__excitement__scallop
  :parameters ()
  :precondition (and (fears_grief_excitement) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_grief_scallop) (not (fears_grief_excitement))))
 (:action overcome__loneliness__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_loneliness_sweetroll) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_loneliness_sweetroll)) (fears_loneliness_lubricity)))
 (:action overcome__loneliness__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_shrimp) (craves_loneliness_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_loneliness_shrimp)) (fears_loneliness_lubricity)))
 (:action overcome__loneliness__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_loneliness_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_loneliness_lubricity) (not (craves_loneliness_cherry))))
 (:action overcome__loneliness__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_loneliness_chocolate) (not (v11_4)) (craves_lubricity_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_loneliness_chocolate)) (fears_loneliness_lubricity)))
 (:action succumb__grief__satisfaction__scallop
  :parameters ()
  :precondition (and (fears_grief_satisfaction) (craves_satisfaction_scallop) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief_scallop) (not (fears_grief_satisfaction))))
 (:action overcome__loneliness__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_loneliness_chocolate) (craves_intoxication_chocolate) (not (v8_4)) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_loneliness_intoxication) (not (craves_loneliness_chocolate))))
 (:action overcome__loneliness__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_loneliness_sweetroll) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_loneliness_intoxication) (not (craves_loneliness_sweetroll))))
 (:action overcome__loneliness__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_shrimp) (craves_loneliness_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_loneliness_intoxication) (not (craves_loneliness_shrimp))))
 (:action overcome__loneliness__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_cherry) (craves_loneliness_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_loneliness_intoxication) (not (craves_loneliness_cherry))))
 (:action succumb__depression__lubricity__sweetroll
  :parameters ()
  :precondition (and (craves_lubricity_sweetroll) (fears_depression_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_depression_sweetroll) (not (fears_depression_lubricity))))
 (:action succumb__depression__lubricity__cherry
  :parameters ()
  :precondition (and (craves_lubricity_cherry) (fears_depression_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_depression_cherry) (not (fears_depression_lubricity))))
 (:action succumb__depression__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (fears_depression_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_depression_lubricity)) (craves_depression_shrimp)))
 (:action succumb__depression__lubricity__scallop
  :parameters ()
  :precondition (and (craves_lubricity_scallop) (fears_depression_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_depression_lubricity)) (craves_depression_scallop)))
 (:action overcome__grief2__excitement__sweetroll
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_sweetroll) (craves_grief2_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief2_excitement) (not (craves_grief2_sweetroll))))
 (:action overcome__grief2__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v2_4)) (craves_grief2_shrimp) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief2_excitement) (not (craves_grief2_shrimp))))
 (:action overcome__grief2__excitement__cherry
  :parameters ()
  :precondition (and (craves_grief2_cherry) (not (v2_4)) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief2_excitement) (not (craves_grief2_cherry))))
 (:action succumb__depression__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_lubricity_broccoli) (fears_depression_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_depression_broccoli) (not (fears_depression_lubricity))))
 (:action overcome__grief2__excitement__scallop
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_scallop) (craves_grief2_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief2_excitement) (not (craves_grief2_scallop))))
 (:action succumb__depression__lubricity__turkey
  :parameters ()
  :precondition (and (craves_lubricity_turkey) (fears_depression_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_depression_turkey) (not (fears_depression_lubricity))))
 (:action succumb__depression__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_depression_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_depression_chocolate) (not (fears_depression_lubricity))))
 (:action overcome__grief2__excitement__chocolate
  :parameters ()
  :precondition (and (craves_grief2_chocolate) (not (v2_4)) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief2_excitement) (not (craves_grief2_chocolate))))
 (:action succumb__depression__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_depression_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_depression_intoxication)) (craves_depression_sweetroll)))
 (:action succumb__depression__intoxication__cherry
  :parameters ()
  :precondition (and (craves_intoxication_cherry) (fears_depression_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_depression_intoxication)) (craves_depression_cherry)))
 (:action overcome__grief2__satisfaction__chocolate
  :parameters ()
  :precondition (and (craves_grief2_chocolate) (not (v1_4)) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_grief2_satisfaction) (not (craves_grief2_chocolate))))
 (:action overcome__grief2__satisfaction__sweetroll
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_sweetroll) (craves_grief2_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief2_sweetroll)) (fears_grief2_satisfaction)))
 (:action overcome__grief2__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v1_4)) (craves_grief2_shrimp) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief2_shrimp)) (fears_grief2_satisfaction)))
 (:action overcome__grief2__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_grief2_cherry) (craves_satisfaction_cherry) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_grief2_satisfaction) (not (craves_grief2_cherry))))
 (:action overcome__grief2__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (not (v1_4)) (craves_grief2_scallop) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_grief2_satisfaction) (not (craves_grief2_scallop))))
 (:action succumb__sciatica__excitement__scallop
  :parameters ()
  :precondition (and (craves_excitement_scallop) (fears_sciatica_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_sciatica_scallop) (not (fears_sciatica_excitement))))
 (:action succumb__sciatica__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_sciatica_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_sciatica_scallop) (not (fears_sciatica_satisfaction))))
 (:action succumb__grief__lubricity__cherry
  :parameters ()
  :precondition (and (fears_grief_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_grief_cherry) (not (fears_grief_lubricity))))
 (:action succumb__grief__lubricity__shrimp
  :parameters ()
  :precondition (and (fears_grief_lubricity) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief_lubricity)) (craves_grief_shrimp)))
 (:action succumb__grief__lubricity__scallop
  :parameters ()
  :precondition (and (fears_grief_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_grief_scallop) (not (fears_grief_lubricity))))
 (:action overcome__loneliness__excitement__chocolate
  :parameters ()
  :precondition (and (craves_loneliness_chocolate) (not (v2_4)) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_loneliness_chocolate)) (fears_loneliness_excitement)))
 (:action succumb__grief__lubricity__broccoli
  :parameters ()
  :precondition (and (fears_grief_lubricity) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief_lubricity)) (craves_grief_broccoli)))
 (:action overcome__loneliness__excitement__sweetroll
  :parameters ()
  :precondition (and (not (v2_4)) (craves_loneliness_sweetroll) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_loneliness_sweetroll)) (fears_loneliness_excitement)))
 (:action succumb__grief__lubricity__turkey
  :parameters ()
  :precondition (and (fears_grief_lubricity) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_grief_turkey) (not (fears_grief_lubricity))))
 (:action overcome__loneliness__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v2_4)) (craves_loneliness_shrimp) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_loneliness_shrimp)) (fears_loneliness_excitement)))
 (:action succumb__grief__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_grief_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief_lubricity)) (craves_grief_chocolate)))
 (:action overcome__loneliness__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_cherry) (craves_loneliness_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_loneliness_cherry)) (fears_loneliness_excitement)))
 (:action succumb__grief__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_grief_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief_lubricity)) (craves_grief_sweetroll)))
 (:action overcome__loneliness__excitement__scallop
  :parameters ()
  :precondition (and (craves_loneliness_scallop) (not (v2_4)) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_loneliness_scallop)) (fears_loneliness_excitement)))
 (:action overcome__loneliness__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (craves_loneliness_scallop) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_loneliness_satisfaction) (not (craves_loneliness_scallop))))
 (:action succumb__grief__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_grief_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_grief_intoxication)) (craves_grief_sweetroll)))
 (:action succumb__grief__intoxication__cherry
  :parameters ()
  :precondition (and (fears_grief_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief_cherry) (not (fears_grief_intoxication))))
 (:action overcome__loneliness__satisfaction__chocolate
  :parameters ()
  :precondition (and (craves_loneliness_chocolate) (not (v1_4)) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_loneliness_satisfaction) (not (craves_loneliness_chocolate))))
 (:action overcome__loneliness__satisfaction__sweetroll
  :parameters ()
  :precondition (and (not (v1_4)) (craves_loneliness_sweetroll) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_loneliness_satisfaction) (not (craves_loneliness_sweetroll))))
 (:action overcome__loneliness__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v1_4)) (craves_loneliness_shrimp) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_loneliness_satisfaction) (not (craves_loneliness_shrimp))))
 (:action overcome__loneliness__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_loneliness_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_loneliness_satisfaction) (not (craves_loneliness_cherry))))
 (:action succumb__angina__lubricity__broccoli
  :parameters ()
  :precondition (and (fears_angina_lubricity) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_angina_lubricity)) (craves_angina_broccoli)))
 (:action succumb__angina__lubricity__turkey
  :parameters ()
  :precondition (and (fears_angina_lubricity) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_angina_turkey) (not (fears_angina_lubricity))))
 (:action overcome__sciatica__excitement__cherry
  :parameters ()
  :precondition (and (craves_sciatica_cherry) (not (v2_4)) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_cherry))))
 (:action overcome__sciatica__excitement__sweetroll
  :parameters ()
  :precondition (and (craves_sciatica_sweetroll) (not (v2_4)) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_sweetroll))))
 (:action succumb__angina__intoxication__broccoli
  :parameters ()
  :precondition (and (fears_angina_intoxication) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_angina_intoxication)) (craves_angina_broccoli)))
 (:action succumb__angina__intoxication__turkey
  :parameters ()
  :precondition (and (fears_angina_intoxication) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_angina_turkey) (not (fears_angina_intoxication))))
 (:action succumb__angina__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_angina_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_angina_intoxication)) (craves_angina_chocolate)))
 (:action succumb__angina__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_angina_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_angina_sweetroll) (not (fears_angina_intoxication))))
 (:action succumb__angina__intoxication__tuna
  :parameters ()
  :precondition (and (fears_angina_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_angina_intoxication)) (craves_angina_tuna)))
 (:action succumb__angina__intoxication__cherry
  :parameters ()
  :precondition (and (fears_angina_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_angina_intoxication)) (craves_angina_cherry)))
 (:action succumb__angina__intoxication__shrimp
  :parameters ()
  :precondition (and (fears_angina_intoxication) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_angina_intoxication)) (craves_angina_shrimp)))
 (:action overcome__sciatica__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_sciatica_sweetroll) (not (v1_4)) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_sciatica_sweetroll)) (fears_sciatica_satisfaction)))
 (:action overcome__sciatica__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_sciatica_cherry) (craves_satisfaction_cherry) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_sciatica_satisfaction) (not (craves_sciatica_cherry))))
 (:action succumb__angina__excitement__scallop
  :parameters ()
  :precondition (and (fears_angina_excitement) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_angina_scallop) (not (fears_angina_excitement))))
 (:action overcome__sciatica__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_broccoli) (craves_sciatica_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_sciatica_broccoli)) (fears_sciatica_lubricity)))
 (:action overcome__sciatica__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_turkey) (craves_sciatica_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_sciatica_turkey)) (fears_sciatica_lubricity)))
 (:action succumb__angina__satisfaction__scallop
  :parameters ()
  :precondition (and (fears_angina_satisfaction) (craves_satisfaction_scallop) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_angina_satisfaction)) (craves_angina_scallop)))
 (:action overcome__sciatica__lubricity__sweetroll
  :parameters ()
  :precondition (and (craves_sciatica_sweetroll) (not (v11_4)) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_sciatica_sweetroll)) (fears_sciatica_lubricity)))
 (:action overcome__sciatica__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_sciatica_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_sciatica_cherry)) (fears_sciatica_lubricity)))
 (:action overcome__sciatica__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_sciatica_cherry) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_sciatica_intoxication) (not (craves_sciatica_cherry))))
 (:action overcome__sciatica__intoxication__sweetroll
  :parameters ()
  :precondition (and (craves_sciatica_sweetroll) (not (v8_4)) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_sciatica_intoxication) (not (craves_sciatica_sweetroll))))
 (:action succumb__abrasion__lubricity__turkey
  :parameters ()
  :precondition (and (craves_lubricity_turkey) (fears_abrasion_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_abrasion_lubricity)) (craves_abrasion_turkey)))
 (:action succumb__abrasion__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (fears_abrasion_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_abrasion_lubricity)) (craves_abrasion_shrimp)))
 (:action succumb__abrasion__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_abrasion_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_abrasion_lubricity)) (craves_abrasion_sweetroll)))
 (:action succumb__abrasion__lubricity__scallop
  :parameters ()
  :precondition (and (fears_abrasion_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_abrasion_lubricity)) (craves_abrasion_scallop)))
 (:action succumb__abrasion__lubricity__cherry
  :parameters ()
  :precondition (and (fears_abrasion_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_abrasion_lubricity)) (craves_abrasion_cherry)))
 (:action succumb__abrasion__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_abrasion_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_abrasion_lubricity)) (craves_abrasion_chocolate)))
 (:action succumb__abrasion__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_lubricity_broccoli) (fears_abrasion_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_abrasion_lubricity)) (craves_abrasion_broccoli)))
 (:action overcome__grief__excitement__sweetroll
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_sweetroll) (craves_grief_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief_excitement) (not (craves_grief_sweetroll))))
 (:action overcome__grief__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_cherry) (craves_grief_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief_excitement) (not (craves_grief_cherry))))
 (:action succumb__abrasion__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_abrasion_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_abrasion_sweetroll) (not (fears_abrasion_intoxication))))
 (:action succumb__abrasion__intoxication__cherry
  :parameters ()
  :precondition (and (fears_abrasion_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_abrasion_intoxication)) (craves_abrasion_cherry)))
 (:action overcome__grief__satisfaction__sweetroll
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_sweetroll) (craves_grief_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_grief_satisfaction) (not (craves_grief_sweetroll))))
 (:action overcome__grief__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_grief_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief_cherry)) (fears_grief_satisfaction)))
 (:action succumb__abrasion__excitement__scallop
  :parameters ()
  :precondition (and (craves_excitement_scallop) (fears_abrasion_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_abrasion_scallop) (not (fears_abrasion_excitement))))
 (:action overcome__grief__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_cherry) (craves_grief_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief_lubricity) (not (craves_grief_cherry))))
 (:action succumb__abrasion__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_abrasion_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_abrasion_satisfaction)) (craves_abrasion_scallop)))
 (:action overcome__grief__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_broccoli) (craves_grief_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief_lubricity) (not (craves_grief_broccoli))))
 (:action overcome__grief__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_turkey) (craves_grief_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief_lubricity) (not (craves_grief_turkey))))
 (:action overcome__grief__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_sweetroll) (craves_grief_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief_lubricity) (not (craves_grief_sweetroll))))
 (:action feast__intoxication__sweetroll__scallop
  :parameters ()
  :precondition (and (not (v9_4)) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (craves_intoxication_scallop) (not (craves_intoxication_sweetroll))))
 (:action feast__intoxication__sweetroll__shrimp
  :parameters ()
  :precondition (and (not (v9_4)) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_13_0) (v9_0)) (when (not (z_13_0)) (not (v9_0))) (when (z_13_1) (v9_1)) (when (not (z_13_1)) (not (v9_1))) (when (z_13_2) (v9_2)) (when (not (z_13_2)) (not (v9_2))) (when (z_13_3) (v9_3)) (when (not (z_13_3)) (not (v9_3))) (when (z_13_4) (v9_4)) (when (not (z_13_4)) (not (v9_4))) (when (and (v9_4) (not (z_13_4))) (of)) (craves_intoxication_shrimp) (not (craves_intoxication_sweetroll))))
 (:action overcome__grief__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_grief_sweetroll) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_grief_intoxication) (not (craves_grief_sweetroll))))
 (:action overcome__grief__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_cherry) (craves_grief_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_grief_cherry)) (fears_grief_intoxication)))
 (:action succumb__grief2__lubricity__broccoli
  :parameters ()
  :precondition (and (fears_grief2_lubricity) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief2_lubricity)) (craves_grief2_broccoli)))
 (:action succumb__grief2__lubricity__turkey
  :parameters ()
  :precondition (and (craves_lubricity_turkey) (fears_grief2_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_grief2_turkey) (not (fears_grief2_lubricity))))
 (:action overcome__depression__excitement__sweetroll
  :parameters ()
  :precondition (and (craves_depression_sweetroll) (not (v2_4)) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_depression_sweetroll)) (fears_depression_excitement)))
 (:action overcome__depression__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_cherry) (craves_depression_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_depression_excitement) (not (craves_depression_cherry))))
 (:action feast__intoxication__cherry__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_cherry) (not (v7_4)) (not (of)))
  :effect (and (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of)) (not (craves_intoxication_cherry)) (craves_intoxication_shrimp)))
 (:action feast__intoxication__cherry__scallop
  :parameters ()
  :precondition (and (craves_intoxication_cherry) (not (v7_4)) (not (of)))
  :effect (and (when (z_9_0) (v7_0)) (when (not (z_9_0)) (not (v7_0))) (when (z_9_1) (v7_1)) (when (not (z_9_1)) (not (v7_1))) (when (z_9_2) (v7_2)) (when (not (z_9_2)) (not (v7_2))) (when (z_9_3) (v7_3)) (when (not (z_9_3)) (not (v7_3))) (when (z_9_4) (v7_4)) (when (not (z_9_4)) (not (v7_4))) (when (and (v7_4) (not (z_9_4))) (of)) (not (craves_intoxication_cherry)) (craves_intoxication_scallop)))
 (:action overcome__depression__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_depression_sweetroll) (not (v1_4)) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_depression_sweetroll)) (fears_depression_satisfaction)))
 (:action overcome__depression__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_depression_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_depression_cherry)) (fears_depression_satisfaction)))
 (:action overcome__depression__lubricity__sweetroll
  :parameters ()
  :precondition (and (craves_depression_sweetroll) (not (v11_4)) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_depression_sweetroll)) (fears_depression_lubricity)))
 (:action overcome__depression__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_depression_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_depression_lubricity) (not (craves_depression_cherry))))
 (:action overcome__depression__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_depression_broccoli) (not (v11_4)) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_depression_lubricity) (not (craves_depression_broccoli))))
 (:action overcome__depression__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_turkey) (craves_depression_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_depression_turkey)) (fears_depression_lubricity)))
 (:action overcome__depression__intoxication__sweetroll
  :parameters ()
  :precondition (and (craves_depression_sweetroll) (not (v8_4)) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_depression_sweetroll)) (fears_depression_intoxication)))
 (:action overcome__depression__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_depression_cherry) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_depression_cherry)) (fears_depression_intoxication)))
 (:action succumb__loneliness__lubricity__turkey
  :parameters ()
  :precondition (and (fears_loneliness_lubricity) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_loneliness_lubricity)) (craves_loneliness_turkey)))
 (:action succumb__laceration__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_lubricity_broccoli) (fears_laceration_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_laceration_lubricity)) (craves_laceration_broccoli)))
 (:action feast__satisfaction__scallop__sweetroll
  :parameters ()
  :precondition (and (not (v10_4)) (craves_satisfaction_scallop) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (craves_satisfaction_sweetroll) (not (craves_satisfaction_scallop))))
 (:action succumb__laceration__lubricity__turkey
  :parameters ()
  :precondition (and (craves_lubricity_turkey) (fears_laceration_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_laceration_turkey) (not (fears_laceration_lubricity))))
 (:action feast__satisfaction__scallop__cherry
  :parameters ()
  :precondition (and (not (v10_4)) (craves_satisfaction_scallop) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (craves_satisfaction_cherry) (not (craves_satisfaction_scallop))))
 (:action overcome__prostatitis__excitement__sweetroll
  :parameters ()
  :precondition (and (not (v2_4)) (craves_prostatitis_sweetroll) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_prostatitis_sweetroll)) (fears_prostatitis_excitement)))
 (:action overcome__prostatitis__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_cherry) (craves_prostatitis_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_prostatitis_excitement) (not (craves_prostatitis_cherry))))
 (:action succumb__laceration__intoxication__cherry
  :parameters ()
  :precondition (and (craves_intoxication_cherry) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_laceration_cherry) (not (fears_laceration_intoxication))))
 (:action succumb__laceration__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_laceration_shrimp) (not (fears_laceration_intoxication))))
 (:action succumb__laceration__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_laceration_intoxication)) (craves_laceration_broccoli)))
 (:action succumb__laceration__intoxication__turkey
  :parameters ()
  :precondition (and (craves_intoxication_turkey) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_laceration_intoxication)) (craves_laceration_turkey)))
 (:action succumb__laceration__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_laceration_chocolate) (not (fears_laceration_intoxication))))
 (:action succumb__laceration__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_laceration_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_laceration_intoxication)) (craves_laceration_sweetroll)))
 (:action succumb__laceration__intoxication__tuna
  :parameters ()
  :precondition (and (fears_laceration_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_laceration_tuna) (not (fears_laceration_intoxication))))
 (:action overcome__prostatitis__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_prostatitis_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_prostatitis_satisfaction) (not (craves_prostatitis_cherry))))
 (:action overcome__prostatitis__satisfaction__sweetroll
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_sweetroll) (craves_prostatitis_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_prostatitis_satisfaction) (not (craves_prostatitis_sweetroll))))
 (:action succumb__laceration__excitement__scallop
  :parameters ()
  :precondition (and (craves_excitement_scallop) (fears_laceration_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_laceration_excitement)) (craves_laceration_scallop)))
 (:action overcome__prostatitis__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_prostatitis_turkey) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_prostatitis_lubricity) (not (craves_prostatitis_turkey))))
 (:action overcome__prostatitis__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_sweetroll) (craves_prostatitis_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_prostatitis_sweetroll)) (fears_prostatitis_lubricity)))
 (:action overcome__prostatitis__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_cherry) (craves_prostatitis_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_prostatitis_lubricity) (not (craves_prostatitis_cherry))))
 (:action succumb__laceration__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_laceration_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_laceration_satisfaction)) (craves_laceration_scallop)))
 (:action overcome__prostatitis__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_prostatitis_broccoli) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_prostatitis_broccoli)) (fears_prostatitis_lubricity)))
 (:action overcome__prostatitis__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_prostatitis_sweetroll) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_prostatitis_intoxication) (not (craves_prostatitis_sweetroll))))
 (:action overcome__prostatitis__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_cherry) (craves_prostatitis_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_prostatitis_intoxication) (not (craves_prostatitis_cherry))))
 (:action overcome__jealousy__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_cherry) (craves_jealousy_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_jealousy_cherry)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__excitement__scallop
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_scallop) (craves_jealousy_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_jealousy_scallop)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v2_4)) (craves_jealousy_broccoli) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_jealousy_broccoli)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__excitement__chocolate
  :parameters ()
  :precondition (and (craves_jealousy_chocolate) (not (v2_4)) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_jealousy_chocolate)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__excitement__turkey
  :parameters ()
  :precondition (and (craves_jealousy_turkey) (craves_excitement_turkey) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_jealousy_turkey)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (craves_jealousy_tuna) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_jealousy_tuna)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__excitement__sweetroll
  :parameters ()
  :precondition (and (not (v2_4)) (craves_jealousy_sweetroll) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_jealousy_sweetroll)) (fears_jealousy_excitement)))
 (:action overcome__jealousy__satisfaction__sweetroll
  :parameters ()
  :precondition (and (not (v1_4)) (craves_jealousy_sweetroll) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_jealousy_satisfaction) (not (craves_jealousy_sweetroll))))
 (:action overcome__jealousy__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_jealousy_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_jealousy_satisfaction) (not (craves_jealousy_cherry))))
 (:action overcome__jealousy__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (not (v1_4)) (craves_jealousy_scallop) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_jealousy_satisfaction) (not (craves_jealousy_scallop))))
 (:action overcome__jealousy__satisfaction__broccoli
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_broccoli) (craves_jealousy_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_jealousy_satisfaction) (not (craves_jealousy_broccoli))))
 (:action overcome__jealousy__satisfaction__chocolate
  :parameters ()
  :precondition (and (craves_jealousy_chocolate) (not (v1_4)) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_jealousy_chocolate)) (fears_jealousy_satisfaction)))
 (:action overcome__jealousy__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_jealousy_turkey) (craves_satisfaction_turkey) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_jealousy_satisfaction) (not (craves_jealousy_turkey))))
 (:action overcome__jealousy__satisfaction__tuna
  :parameters ()
  :precondition (and (craves_jealousy_tuna) (not (v1_4)) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_jealousy_satisfaction) (not (craves_jealousy_tuna))))
 (:action overcome__jealousy__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_jealousy_scallop) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_jealousy_lubricity) (not (craves_jealousy_scallop))))
 (:action overcome__jealousy__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_broccoli) (craves_jealousy_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_jealousy_lubricity) (not (craves_jealousy_broccoli))))
 (:action overcome__jealousy__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_jealousy_chocolate) (not (v11_4)) (craves_lubricity_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_jealousy_chocolate)) (fears_jealousy_lubricity)))
 (:action overcome__jealousy__lubricity__turkey
  :parameters ()
  :precondition (and (craves_jealousy_turkey) (not (v11_4)) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_jealousy_lubricity) (not (craves_jealousy_turkey))))
 (:action overcome__jealousy__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_jealousy_sweetroll) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_jealousy_lubricity) (not (craves_jealousy_sweetroll))))
 (:action overcome__jealousy__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_jealousy_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_jealousy_lubricity) (not (craves_jealousy_cherry))))
 (:action overcome__jealousy__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_jealousy_cherry) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_jealousy_intoxication) (not (craves_jealousy_cherry))))
 (:action overcome__jealousy__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_broccoli) (craves_jealousy_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_jealousy_intoxication) (not (craves_jealousy_broccoli))))
 (:action overcome__jealousy__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_jealousy_chocolate) (craves_intoxication_chocolate) (not (v8_4)) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_jealousy_intoxication) (not (craves_jealousy_chocolate))))
 (:action overcome__jealousy__intoxication__turkey
  :parameters ()
  :precondition (and (craves_jealousy_turkey) (not (v8_4)) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_jealousy_intoxication) (not (craves_jealousy_turkey))))
 (:action overcome__jealousy__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_jealousy_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_jealousy_intoxication) (not (craves_jealousy_tuna))))
 (:action overcome__jealousy__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_jealousy_sweetroll) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_jealousy_intoxication) (not (craves_jealousy_sweetroll))))
 (:action succumb__hangover__lubricity__turkey
  :parameters ()
  :precondition (and (craves_lubricity_turkey) (fears_hangover_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_hangover_turkey) (not (fears_hangover_lubricity))))
 (:action succumb__hangover__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_lubricity_broccoli) (fears_hangover_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_hangover_broccoli) (not (fears_hangover_lubricity))))
 (:action overcome__anger__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_anger_cherry) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anger_cherry)) (fears_anger_intoxication)))
 (:action succumb__boils__lubricity__shrimp
  :parameters ()
  :precondition (and (fears_boils_lubricity) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_boils_lubricity)) (craves_boils_shrimp)))
 (:action succumb__boils__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_boils_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_boils_lubricity)) (craves_boils_sweetroll)))
 (:action succumb__boils__lubricity__scallop
  :parameters ()
  :precondition (and (fears_boils_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_boils_lubricity)) (craves_boils_scallop)))
 (:action succumb__boils__lubricity__cherry
  :parameters ()
  :precondition (and (fears_boils_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_boils_cherry) (not (fears_boils_lubricity))))
 (:action overcome__dread__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_dread_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_dread_excitement) (not (craves_dread_chocolate))))
 (:action overcome__dread__excitement__sweetroll
  :parameters ()
  :precondition (and (not (v2_4)) (craves_dread_sweetroll) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_dread_excitement) (not (craves_dread_sweetroll))))
 (:action overcome__dread__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v2_4)) (craves_dread_shrimp) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_dread_excitement) (not (craves_dread_shrimp))))
 (:action succumb__boils__lubricity__chocolate
  :parameters ()
  :precondition (and (fears_boils_lubricity) (craves_lubricity_chocolate) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_boils_lubricity)) (craves_boils_chocolate)))
 (:action overcome__dread__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_cherry) (craves_dread_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_dread_cherry)) (fears_dread_excitement)))
 (:action succumb__boils__lubricity__broccoli
  :parameters ()
  :precondition (and (fears_boils_lubricity) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_boils_broccoli) (not (fears_boils_lubricity))))
 (:action overcome__dread__excitement__scallop
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_scallop) (craves_dread_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_dread_scallop)) (fears_dread_excitement)))
 (:action succumb__boils__lubricity__turkey
  :parameters ()
  :precondition (and (fears_boils_lubricity) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_boils_lubricity)) (craves_boils_turkey)))
 (:action succumb__boils__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_boils_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_sweetroll)))
 (:action succumb__boils__intoxication__cherry
  :parameters ()
  :precondition (and (fears_boils_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_boils_cherry) (not (fears_boils_intoxication))))
 (:action overcome__dread__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_dread_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_dread_satisfaction) (not (craves_dread_chocolate))))
 (:action overcome__dread__satisfaction__sweetroll
  :parameters ()
  :precondition (and (not (v1_4)) (craves_dread_sweetroll) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_dread_satisfaction) (not (craves_dread_sweetroll))))
 (:action overcome__dread__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v1_4)) (craves_dread_shrimp) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_dread_shrimp)) (fears_dread_satisfaction)))
 (:action overcome__dread__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_dread_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_dread_cherry)) (fears_dread_satisfaction)))
 (:action overcome__dread__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (not (v1_4)) (craves_dread_scallop) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_dread_scallop)) (fears_dread_satisfaction)))
 (:action succumb__boils__excitement__scallop
  :parameters ()
  :precondition (and (fears_boils_excitement) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_boils_excitement)) (craves_boils_scallop)))
 (:action overcome__dread__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_dread_sweetroll) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_dread_lubricity) (not (craves_dread_sweetroll))))
 (:action overcome__dread__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_shrimp) (craves_dread_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_dread_shrimp)) (fears_dread_lubricity)))
 (:action overcome__dread__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_dread_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_dread_cherry)) (fears_dread_lubricity)))
 (:action overcome__dread__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_dread_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_dread_lubricity) (not (craves_dread_chocolate))))
 (:action succumb__boils__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_boils_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_boils_satisfaction)) (craves_boils_scallop)))
 (:action overcome__dread__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_dread_sweetroll) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_dread_intoxication) (not (craves_dread_sweetroll))))
 (:action overcome__dread__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_shrimp) (craves_dread_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_dread_intoxication) (not (craves_dread_shrimp))))
 (:action overcome__dread__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_cherry) (craves_dread_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_dread_cherry)) (fears_dread_intoxication)))
 (:action overcome__dread__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_dread_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_dread_intoxication) (not (craves_dread_chocolate))))
 (:action overcome__hangover__excitement__scallop
  :parameters ()
  :precondition (and (craves_hangover_scallop) (not (v2_4)) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_hangover_scallop)) (fears_hangover_excitement)))
 (:action succumb__anxiety__lubricity__scallop
  :parameters ()
  :precondition (and (fears_anxiety_lubricity) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_anxiety_scallop) (not (fears_anxiety_lubricity))))
 (:action succumb__anxiety__lubricity__cherry
  :parameters ()
  :precondition (and (fears_anxiety_lubricity) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_anxiety_cherry) (not (fears_anxiety_lubricity))))
 (:action overcome__hangover__excitement__chocolate
  :parameters ()
  :precondition (and (craves_hangover_chocolate) (not (v2_4)) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_hangover_chocolate)) (fears_hangover_excitement)))
 (:action succumb__anxiety__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_lubricity_chocolate) (fears_anxiety_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_anxiety_lubricity)) (craves_anxiety_chocolate)))
 (:action succumb__anxiety__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_lubricity_broccoli) (fears_anxiety_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_anxiety_lubricity)) (craves_anxiety_broccoli)))
 (:action overcome__hangover__excitement__sweetroll
  :parameters ()
  :precondition (and (craves_hangover_sweetroll) (not (v2_4)) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_hangover_sweetroll)) (fears_hangover_excitement)))
 (:action succumb__anxiety__lubricity__turkey
  :parameters ()
  :precondition (and (craves_lubricity_turkey) (fears_anxiety_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_anxiety_turkey) (not (fears_anxiety_lubricity))))
 (:action overcome__hangover__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (not (v2_4)) (craves_hangover_shrimp) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_hangover_excitement) (not (craves_hangover_shrimp))))
 (:action succumb__anxiety__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_lubricity_shrimp) (fears_anxiety_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_anxiety_shrimp) (not (fears_anxiety_lubricity))))
 (:action overcome__hangover__excitement__cherry
  :parameters ()
  :precondition (and (not (v2_4)) (craves_hangover_cherry) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_hangover_cherry)) (fears_hangover_excitement)))
 (:action succumb__anxiety__lubricity__sweetroll
  :parameters ()
  :precondition (and (fears_anxiety_lubricity) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_anxiety_sweetroll) (not (fears_anxiety_lubricity))))
 (:action overcome__hangover__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (not (v1_4)) (craves_hangover_cherry) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_hangover_satisfaction) (not (craves_hangover_cherry))))
 (:action succumb__anxiety__intoxication__cherry
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_cherry)))
 (:action overcome__hangover__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_hangover_scallop) (craves_satisfaction_scallop) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_hangover_satisfaction) (not (craves_hangover_scallop))))
 (:action overcome__hangover__satisfaction__chocolate
  :parameters ()
  :precondition (and (craves_hangover_chocolate) (not (v1_4)) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_hangover_chocolate)) (fears_hangover_satisfaction)))
 (:action overcome__hangover__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_hangover_sweetroll) (not (v1_4)) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_hangover_sweetroll)) (fears_hangover_satisfaction)))
 (:action succumb__anxiety__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_anxiety_intoxication)) (craves_anxiety_sweetroll)))
 (:action overcome__hangover__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (not (v1_4)) (craves_hangover_shrimp) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_hangover_satisfaction) (not (craves_hangover_shrimp))))
 (:action succumb__anxiety__excitement__scallop
  :parameters ()
  :precondition (and (craves_excitement_scallop) (fears_anxiety_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_anxiety_scallop) (not (fears_anxiety_excitement))))
 (:action overcome__hangover__lubricity__chocolate
  :parameters ()
  :precondition (and (craves_hangover_chocolate) (not (v11_4)) (craves_lubricity_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_hangover_chocolate)) (fears_hangover_lubricity)))
 (:action overcome__hangover__lubricity__sweetroll
  :parameters ()
  :precondition (and (not (v11_4)) (craves_hangover_sweetroll) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_hangover_sweetroll)) (fears_hangover_lubricity)))
 (:action overcome__hangover__lubricity__shrimp
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_shrimp) (craves_hangover_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_hangover_shrimp)) (fears_hangover_lubricity)))
 (:action overcome__hangover__lubricity__cherry
  :parameters ()
  :precondition (and (not (v11_4)) (craves_hangover_cherry) (craves_lubricity_cherry) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_hangover_cherry)) (fears_hangover_lubricity)))
 (:action succumb__anxiety__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_anxiety_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_anxiety_scallop) (not (fears_anxiety_satisfaction))))
 (:action overcome__hangover__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (craves_hangover_chocolate) (not (v8_4)) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_hangover_chocolate)) (fears_hangover_intoxication)))
 (:action overcome__hangover__intoxication__sweetroll
  :parameters ()
  :precondition (and (not (v8_4)) (craves_hangover_sweetroll) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_sweetroll))))
 (:action overcome__hangover__intoxication__shrimp
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_shrimp) (craves_hangover_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_shrimp))))
 (:action overcome__hangover__intoxication__cherry
  :parameters ()
  :precondition (and (not (v8_4)) (craves_hangover_cherry) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_cherry))))
 (:action overcome__anxiety__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_anxiety_scallop) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_anxiety_scallop)) (fears_anxiety_lubricity)))
 (:action overcome__anxiety__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_anxiety_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anxiety_lubricity) (not (craves_anxiety_tuna))))
 (:action overcome__anxiety__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_anxiety_scallop) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anxiety_scallop)) (fears_anxiety_intoxication)))
 (:action succumb__jealousy__lubricity__tuna
  :parameters ()
  :precondition (and (fears_jealousy_lubricity) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_jealousy_tuna) (not (fears_jealousy_lubricity))))
 (:action feast__lubricity__tuna__turkey
  :parameters ()
  :precondition (and (not (v5_4)) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (craves_lubricity_turkey) (not (craves_lubricity_tuna))))
 (:action feast__lubricity__tuna__broccoli
  :parameters ()
  :precondition (and (not (v5_4)) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_5_0) (v5_0)) (when (not (z_5_0)) (not (v5_0))) (when (z_5_1) (v5_1)) (when (not (z_5_1)) (not (v5_1))) (when (z_5_2) (v5_2)) (when (not (z_5_2)) (not (v5_2))) (when (z_5_3) (v5_3)) (when (not (z_5_3)) (not (v5_3))) (when (z_5_4) (v5_4)) (when (not (z_5_4)) (not (v5_4))) (when (and (v5_4) (not (z_5_4))) (of)) (not (craves_lubricity_tuna)) (craves_lubricity_broccoli)))
 (:action succumb__jealousy__intoxication__scallop
  :parameters ()
  :precondition (and (craves_intoxication_scallop) (fears_jealousy_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_jealousy_intoxication)) (craves_jealousy_scallop)))
 (:action overcome__boils__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_scallop) (craves_boils_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_boils_scallop)) (fears_boils_lubricity)))
 (:action overcome__boils__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_boils_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_boils_tuna)) (fears_boils_lubricity)))
 (:action overcome__boils__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_scallop) (craves_boils_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_boils_scallop)) (fears_boils_intoxication)))
 (:action succumb__prostatitis__lubricity__tuna
  :parameters ()
  :precondition (and (fears_prostatitis_lubricity) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_prostatitis_tuna) (not (fears_prostatitis_lubricity))))
 (:action overcome__anxiety__excitement__scallop
  :parameters ()
  :precondition (and (craves_anxiety_scallop) (not (v2_4)) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_anxiety_scallop)) (fears_anxiety_excitement)))
 (:action succumb__prostatitis__intoxication__scallop
  :parameters ()
  :precondition (and (fears_prostatitis_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_prostatitis_scallop) (not (fears_prostatitis_intoxication))))
 (:action overcome__anxiety__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (craves_anxiety_scallop) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_anxiety_scallop)) (fears_anxiety_satisfaction)))
 (:action overcome__abrasion__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_abrasion_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_abrasion_lubricity) (not (craves_abrasion_tuna))))
 (:action overcome__abrasion__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_abrasion_scallop) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_abrasion_scallop)) (fears_abrasion_lubricity)))
 (:action overcome__abrasion__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_abrasion_scallop) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_abrasion_scallop)) (fears_abrasion_intoxication)))
 (:action succumb__dread__lubricity__tuna
  :parameters ()
  :precondition (and (fears_dread_lubricity) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_dread_lubricity)) (craves_dread_tuna)))
 (:action overcome__boils__excitement__scallop
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_scallop) (craves_boils_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_boils_scallop)) (fears_boils_excitement)))
 (:action succumb__dread__intoxication__tuna
  :parameters ()
  :precondition (and (fears_dread_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_dread_intoxication)) (craves_dread_tuna)))
 (:action succumb__dread__intoxication__turkey
  :parameters ()
  :precondition (and (fears_dread_intoxication) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_dread_turkey) (not (fears_dread_intoxication))))
 (:action succumb__dread__intoxication__shrimp
  :parameters ()
  :precondition (and (fears_dread_intoxication) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_dread_shrimp) (not (fears_dread_intoxication))))
 (:action succumb__dread__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_dread_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_dread_intoxication)) (craves_dread_sweetroll)))
 (:action succumb__dread__intoxication__scallop
  :parameters ()
  :precondition (and (fears_dread_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_dread_scallop) (not (fears_dread_intoxication))))
 (:action succumb__dread__intoxication__cherry
  :parameters ()
  :precondition (and (fears_dread_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_dread_intoxication)) (craves_dread_cherry)))
 (:action succumb__dread__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_dread_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_dread_intoxication)) (craves_dread_chocolate)))
 (:action succumb__dread__intoxication__broccoli
  :parameters ()
  :precondition (and (fears_dread_intoxication) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_dread_intoxication)) (craves_dread_broccoli)))
 (:action overcome__boils__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (not (v1_4)) (craves_boils_scallop) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_boils_scallop)) (fears_boils_satisfaction)))
 (:action succumb__dread__excitement__turkey
  :parameters ()
  :precondition (and (fears_dread_excitement) (craves_excitement_turkey) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_dread_turkey) (not (fears_dread_excitement))))
 (:action succumb__dread__excitement__chocolate
  :parameters ()
  :precondition (and (fears_dread_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_dread_excitement)) (craves_dread_chocolate)))
 (:action succumb__dread__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_dread_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_dread_excitement)) (craves_dread_sweetroll)))
 (:action succumb__dread__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_dread_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_dread_excitement)) (craves_dread_tuna)))
 (:action succumb__dread__excitement__cherry
  :parameters ()
  :precondition (and (fears_dread_excitement) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_dread_excitement)) (craves_dread_cherry)))
 (:action succumb__dread__excitement__shrimp
  :parameters ()
  :precondition (and (fears_dread_excitement) (craves_excitement_shrimp) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_dread_shrimp) (not (fears_dread_excitement))))
 (:action succumb__dread__excitement__scallop
  :parameters ()
  :precondition (and (fears_dread_excitement) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_dread_scallop) (not (fears_dread_excitement))))
 (:action overcome__angina__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_angina_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_angina_tuna)) (fears_angina_lubricity)))
 (:action overcome__angina__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_scallop) (craves_angina_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_angina_intoxication) (not (craves_angina_scallop))))
 (:action succumb__anger__lubricity__tuna
  :parameters ()
  :precondition (and (craves_lubricity_tuna) (fears_anger_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_anger_lubricity)) (craves_anger_tuna)))
 (:action overcome__abrasion__excitement__scallop
  :parameters ()
  :precondition (and (craves_abrasion_scallop) (not (v2_4)) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_abrasion_scallop)) (fears_abrasion_excitement)))
 (:action succumb__anger__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (fears_anger_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_anger_shrimp) (not (fears_anger_intoxication))))
 (:action succumb__anger__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_anger_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_anger_sweetroll) (not (fears_anger_intoxication))))
 (:action succumb__anger__intoxication__scallop
  :parameters ()
  :precondition (and (fears_anger_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_anger_intoxication)) (craves_anger_scallop)))
 (:action succumb__anger__intoxication__cherry
  :parameters ()
  :precondition (and (fears_anger_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_anger_cherry) (not (fears_anger_intoxication))))
 (:action succumb__anger__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_anger_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_anger_chocolate) (not (fears_anger_intoxication))))
 (:action succumb__anger__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (fears_anger_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_anger_intoxication)) (craves_anger_broccoli)))
 (:action succumb__anger__intoxication__tuna
  :parameters ()
  :precondition (and (fears_anger_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_anger_intoxication)) (craves_anger_tuna)))
 (:action succumb__anger__intoxication__turkey
  :parameters ()
  :precondition (and (craves_intoxication_turkey) (fears_anger_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_anger_turkey) (not (fears_anger_intoxication))))
 (:action overcome__abrasion__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (craves_abrasion_scallop) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_abrasion_scallop)) (fears_abrasion_satisfaction)))
 (:action overcome__laceration__lubricity__tuna
  :parameters ()
  :precondition (and (craves_laceration_tuna) (not (v11_4)) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_laceration_tuna)) (fears_laceration_lubricity)))
 (:action overcome__laceration__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_laceration_scallop) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_laceration_scallop)) (fears_laceration_intoxication)))
 (:action succumb__sciatica__lubricity__tuna
  :parameters ()
  :precondition (and (craves_lubricity_tuna) (fears_sciatica_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_sciatica_lubricity)) (craves_sciatica_tuna)))
 (:action succumb__sciatica__intoxication__scallop
  :parameters ()
  :precondition (and (craves_intoxication_scallop) (fears_sciatica_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_sciatica_scallop) (not (fears_sciatica_intoxication))))
 (:action overcome__grief2__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_grief2_broccoli) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief2_lubricity) (not (craves_grief2_broccoli))))
 (:action overcome__grief2__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_turkey) (craves_grief2_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief2_lubricity) (not (craves_grief2_turkey))))
 (:action succumb__loneliness__intoxication__turkey
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_turkey)))
 (:action succumb__loneliness__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_loneliness_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_chocolate)))
 (:action succumb__loneliness__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_sweetroll)))
 (:action succumb__loneliness__intoxication__tuna
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_loneliness_tuna) (not (fears_loneliness_intoxication))))
 (:action succumb__loneliness__intoxication__cherry
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_cherry)))
 (:action succumb__loneliness__intoxication__shrimp
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_shrimp)))
 (:action succumb__loneliness__intoxication__scallop
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_loneliness_intoxication)) (craves_loneliness_scallop)))
 (:action succumb__loneliness__intoxication__broccoli
  :parameters ()
  :precondition (and (fears_loneliness_intoxication) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_loneliness_broccoli) (not (fears_loneliness_intoxication))))
 (:action overcome__grief2__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_scallop) (craves_grief2_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_grief2_intoxication) (not (craves_grief2_scallop))))
 (:action overcome__grief2__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_broccoli) (craves_grief2_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_grief2_intoxication) (not (craves_grief2_broccoli))))
 (:action overcome__grief2__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_grief2_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_grief2_turkey)) (fears_grief2_intoxication)))
 (:action succumb__loneliness__excitement__chocolate
  :parameters ()
  :precondition (and (fears_loneliness_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_loneliness_excitement)) (craves_loneliness_chocolate)))
 (:action succumb__loneliness__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_loneliness_broccoli) (not (fears_loneliness_excitement))))
 (:action succumb__loneliness__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_loneliness_excitement)) (craves_loneliness_tuna)))
 (:action succumb__loneliness__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_loneliness_excitement)) (craves_loneliness_turkey)))
 (:action succumb__loneliness__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_loneliness_excitement)) (craves_loneliness_shrimp)))
 (:action succumb__loneliness__excitement__sweetroll
  :parameters ()
  :precondition (and (craves_excitement_sweetroll) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_loneliness_excitement)) (craves_loneliness_sweetroll)))
 (:action succumb__loneliness__excitement__scallop
  :parameters ()
  :precondition (and (craves_excitement_scallop) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_loneliness_excitement)) (craves_loneliness_scallop)))
 (:action succumb__loneliness__excitement__cherry
  :parameters ()
  :precondition (and (craves_excitement_cherry) (fears_loneliness_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_loneliness_excitement)) (craves_loneliness_cherry)))
 (:action succumb__loneliness__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_loneliness_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_loneliness_satisfaction)) (craves_loneliness_scallop)))
 (:action succumb__loneliness__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_satisfaction_broccoli) (fears_loneliness_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_loneliness_broccoli) (not (fears_loneliness_satisfaction))))
 (:action succumb__loneliness__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_loneliness_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_loneliness_satisfaction)) (craves_loneliness_turkey)))
 (:action succumb__loneliness__satisfaction__chocolate
  :parameters ()
  :precondition (and (craves_satisfaction_chocolate) (fears_loneliness_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_loneliness_satisfaction)) (craves_loneliness_chocolate)))
 (:action succumb__loneliness__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_satisfaction_sweetroll) (fears_loneliness_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_loneliness_satisfaction)) (craves_loneliness_sweetroll)))
 (:action succumb__loneliness__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_loneliness_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_loneliness_tuna) (not (fears_loneliness_satisfaction))))
 (:action succumb__loneliness__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_loneliness_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_loneliness_satisfaction)) (craves_loneliness_cherry)))
 (:action succumb__loneliness__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_loneliness_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_loneliness_shrimp) (not (fears_loneliness_satisfaction))))
 (:action overcome__loneliness__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_loneliness_turkey) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_loneliness_turkey)) (fears_loneliness_lubricity)))
 (:action overcome__loneliness__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_loneliness_broccoli) (not (v11_4)) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_loneliness_broccoli)) (fears_loneliness_lubricity)))
 (:action overcome__loneliness__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_loneliness_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_loneliness_intoxication) (not (craves_loneliness_turkey))))
 (:action overcome__loneliness__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_loneliness_scallop) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_loneliness_intoxication) (not (craves_loneliness_scallop))))
 (:action overcome__loneliness__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_loneliness_broccoli) (not (v8_4)) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_loneliness_intoxication) (not (craves_loneliness_broccoli))))
 (:action succumb__depression__lubricity__tuna
  :parameters ()
  :precondition (and (craves_lubricity_tuna) (fears_depression_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_depression_lubricity)) (craves_depression_tuna)))
 (:action overcome__grief2__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_grief2_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_grief2_turkey)) (fears_grief2_excitement)))
 (:action overcome__grief2__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v2_4)) (craves_grief2_broccoli) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief2_excitement) (not (craves_grief2_broccoli))))
 (:action succumb__depression__intoxication__scallop
  :parameters ()
  :precondition (and (craves_intoxication_scallop) (fears_depression_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_depression_intoxication)) (craves_depression_scallop)))
 (:action overcome__grief2__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_grief2_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief2_turkey)) (fears_grief2_satisfaction)))
 (:action overcome__grief2__satisfaction__broccoli
  :parameters ()
  :precondition (and (not (v1_4)) (craves_grief2_broccoli) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_grief2_satisfaction) (not (craves_grief2_broccoli))))
 (:action overcome__loneliness__excitement__broccoli
  :parameters ()
  :precondition (and (craves_loneliness_broccoli) (craves_excitement_broccoli) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_loneliness_broccoli)) (fears_loneliness_excitement)))
 (:action overcome__loneliness__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_loneliness_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_loneliness_turkey)) (fears_loneliness_excitement)))
 (:action succumb__grief__lubricity__tuna
  :parameters ()
  :precondition (and (fears_grief_lubricity) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_grief_lubricity)) (craves_grief_tuna)))
 (:action succumb__grief__intoxication__scallop
  :parameters ()
  :precondition (and (fears_grief_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief_scallop) (not (fears_grief_intoxication))))
 (:action overcome__loneliness__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_loneliness_broccoli) (not (v1_4)) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_loneliness_broccoli)) (fears_loneliness_satisfaction)))
 (:action overcome__loneliness__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_loneliness_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_loneliness_satisfaction) (not (craves_loneliness_turkey))))
 (:action succumb__angina__lubricity__tuna
  :parameters ()
  :precondition (and (fears_angina_lubricity) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_angina_lubricity)) (craves_angina_tuna)))
 (:action overcome__sciatica__excitement__scallop
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_scallop) (craves_sciatica_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_sciatica_excitement) (not (craves_sciatica_scallop))))
 (:action succumb__angina__intoxication__scallop
  :parameters ()
  :precondition (and (fears_angina_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_angina_scallop) (not (fears_angina_intoxication))))
 (:action overcome__sciatica__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (not (v1_4)) (craves_sciatica_scallop) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_sciatica_scallop)) (fears_sciatica_satisfaction)))
 (:action overcome__sciatica__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_sciatica_scallop) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_sciatica_scallop)) (fears_sciatica_lubricity)))
 (:action overcome__sciatica__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_sciatica_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_sciatica_tuna)) (fears_sciatica_lubricity)))
 (:action overcome__sciatica__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_sciatica_scallop) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_sciatica_intoxication) (not (craves_sciatica_scallop))))
 (:action succumb__abrasion__lubricity__tuna
  :parameters ()
  :precondition (and (craves_lubricity_tuna) (fears_abrasion_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (not (fears_abrasion_lubricity)) (craves_abrasion_tuna)))
 (:action overcome__grief__excitement__scallop
  :parameters ()
  :precondition (and (craves_grief_scallop) (not (v2_4)) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_grief_excitement) (not (craves_grief_scallop))))
 (:action succumb__abrasion__intoxication__scallop
  :parameters ()
  :precondition (and (fears_abrasion_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_abrasion_scallop) (not (fears_abrasion_intoxication))))
 (:action overcome__grief__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_grief_scallop) (craves_satisfaction_scallop) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief_scallop)) (fears_grief_satisfaction)))
 (:action overcome__grief__lubricity__scallop
  :parameters ()
  :precondition (and (craves_grief_scallop) (not (v11_4)) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief_lubricity) (not (craves_grief_scallop))))
 (:action overcome__grief__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_grief_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief_lubricity) (not (craves_grief_tuna))))
 (:action overcome__grief__intoxication__scallop
  :parameters ()
  :precondition (and (craves_grief_scallop) (not (v8_4)) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_grief_scallop)) (fears_grief_intoxication)))
 (:action succumb__grief2__lubricity__tuna
  :parameters ()
  :precondition (and (craves_lubricity_tuna) (fears_grief2_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_grief2_tuna) (not (fears_grief2_lubricity))))
 (:action feast__intoxication__scallop__cherry
  :parameters ()
  :precondition (and (not (v10_4)) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (not (craves_intoxication_scallop)) (craves_intoxication_cherry)))
 (:action feast__intoxication__scallop__sweetroll
  :parameters ()
  :precondition (and (not (v10_4)) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_17_0) (v10_0)) (when (not (z_17_0)) (not (v10_0))) (when (z_17_1) (v10_1)) (when (not (z_17_1)) (not (v10_1))) (when (z_17_2) (v10_2)) (when (not (z_17_2)) (not (v10_2))) (when (z_17_3) (v10_3)) (when (not (z_17_3)) (not (v10_3))) (when (z_17_4) (v10_4)) (when (not (z_17_4)) (not (v10_4))) (when (and (v10_4) (not (z_17_4))) (of)) (not (craves_intoxication_scallop)) (craves_intoxication_sweetroll)))
 (:action overcome__depression__excitement__scallop
  :parameters ()
  :precondition (and (not (v2_4)) (craves_excitement_scallop) (craves_depression_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_depression_scallop)) (fears_depression_excitement)))
 (:action succumb__grief2__intoxication__sweetroll
  :parameters ()
  :precondition (and (fears_grief2_intoxication) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief2_sweetroll) (not (fears_grief2_intoxication))))
 (:action succumb__grief2__intoxication__tuna
  :parameters ()
  :precondition (and (fears_grief2_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief2_tuna) (not (fears_grief2_intoxication))))
 (:action succumb__grief2__intoxication__cherry
  :parameters ()
  :precondition (and (fears_grief2_intoxication) (craves_intoxication_cherry) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief2_cherry) (not (fears_grief2_intoxication))))
 (:action succumb__grief2__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (fears_grief2_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief2_shrimp) (not (fears_grief2_intoxication))))
 (:action succumb__grief2__intoxication__scallop
  :parameters ()
  :precondition (and (fears_grief2_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief2_scallop) (not (fears_grief2_intoxication))))
 (:action succumb__grief2__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (fears_grief2_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief2_broccoli) (not (fears_grief2_intoxication))))
 (:action succumb__grief2__intoxication__turkey
  :parameters ()
  :precondition (and (craves_intoxication_turkey) (fears_grief2_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief2_turkey) (not (fears_grief2_intoxication))))
 (:action succumb__grief2__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_grief2_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_grief2_chocolate) (not (fears_grief2_intoxication))))
 (:action overcome__depression__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (not (v1_4)) (craves_depression_scallop) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_depression_scallop)) (fears_depression_satisfaction)))
 (:action succumb__grief2__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_grief2_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_grief2_tuna) (not (fears_grief2_excitement))))
 (:action succumb__grief2__excitement__turkey
  :parameters ()
  :precondition (and (fears_grief2_excitement) (craves_excitement_turkey) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_grief2_turkey) (not (fears_grief2_excitement))))
 (:action succumb__grief2__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_grief2_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_grief2_shrimp) (not (fears_grief2_excitement))))
 (:action succumb__grief2__excitement__sweetroll
  :parameters ()
  :precondition (and (fears_grief2_excitement) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_grief2_excitement)) (craves_grief2_sweetroll)))
 (:action succumb__grief2__excitement__scallop
  :parameters ()
  :precondition (and (fears_grief2_excitement) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_grief2_excitement)) (craves_grief2_scallop)))
 (:action succumb__grief2__excitement__cherry
  :parameters ()
  :precondition (and (fears_grief2_excitement) (craves_excitement_cherry) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_grief2_excitement)) (craves_grief2_cherry)))
 (:action succumb__grief2__excitement__chocolate
  :parameters ()
  :precondition (and (fears_grief2_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_grief2_excitement)) (craves_grief2_chocolate)))
 (:action succumb__grief2__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_grief2_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_grief2_excitement)) (craves_grief2_broccoli)))
 (:action overcome__depression__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_depression_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_depression_lubricity) (not (craves_depression_tuna))))
 (:action overcome__depression__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_scallop) (craves_depression_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_depression_scallop)) (fears_depression_lubricity)))
 (:action succumb__grief2__satisfaction__turkey
  :parameters ()
  :precondition (and (fears_grief2_satisfaction) (craves_satisfaction_turkey) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief2_turkey) (not (fears_grief2_satisfaction))))
 (:action succumb__grief2__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_grief2_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief2_chocolate) (not (fears_grief2_satisfaction))))
 (:action succumb__grief2__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_grief2_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief2_sweetroll) (not (fears_grief2_satisfaction))))
 (:action succumb__grief2__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_grief2_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief2_tuna) (not (fears_grief2_satisfaction))))
 (:action succumb__grief2__satisfaction__cherry
  :parameters ()
  :precondition (and (fears_grief2_satisfaction) (craves_satisfaction_cherry) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_grief2_satisfaction)) (craves_grief2_cherry)))
 (:action succumb__grief2__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_grief2_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_grief2_shrimp) (not (fears_grief2_satisfaction))))
 (:action succumb__grief2__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_grief2_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_grief2_satisfaction)) (craves_grief2_scallop)))
 (:action succumb__grief2__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_grief2_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_grief2_satisfaction)) (craves_grief2_broccoli)))
 (:action overcome__depression__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_scallop) (craves_depression_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_depression_scallop)) (fears_depression_intoxication)))
 (:action succumb__loneliness__lubricity__tuna
  :parameters ()
  :precondition (and (fears_loneliness_lubricity) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_loneliness_tuna) (not (fears_loneliness_lubricity))))
 (:action succumb__laceration__lubricity__tuna
  :parameters ()
  :precondition (and (craves_lubricity_tuna) (fears_laceration_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_laceration_tuna) (not (fears_laceration_lubricity))))
 (:action overcome__prostatitis__excitement__scallop
  :parameters ()
  :precondition (and (craves_prostatitis_scallop) (not (v2_4)) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_prostatitis_scallop)) (fears_prostatitis_excitement)))
 (:action succumb__laceration__intoxication__scallop
  :parameters ()
  :precondition (and (craves_intoxication_scallop) (fears_laceration_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_laceration_intoxication)) (craves_laceration_scallop)))
 (:action overcome__prostatitis__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_prostatitis_scallop) (craves_satisfaction_scallop) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_prostatitis_satisfaction) (not (craves_prostatitis_scallop))))
 (:action overcome__prostatitis__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_prostatitis_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_prostatitis_tuna)) (fears_prostatitis_lubricity)))
 (:action overcome__prostatitis__lubricity__scallop
  :parameters ()
  :precondition (and (craves_prostatitis_scallop) (not (v11_4)) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_prostatitis_scallop)) (fears_prostatitis_lubricity)))
 (:action overcome__prostatitis__intoxication__scallop
  :parameters ()
  :precondition (and (craves_prostatitis_scallop) (not (v8_4)) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_prostatitis_intoxication) (not (craves_prostatitis_scallop))))
 (:action succumb__hangover__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (fears_hangover_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_hangover_scallop) (not (fears_hangover_satisfaction))))
 (:action succumb__hangover__satisfaction__cherry
  :parameters ()
  :precondition (and (craves_satisfaction_cherry) (fears_hangover_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_hangover_cherry) (not (fears_hangover_satisfaction))))
 (:action succumb__hangover__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_hangover_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_hangover_satisfaction)) (craves_hangover_chocolate)))
 (:action succumb__hangover__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_satisfaction_broccoli) (fears_hangover_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_hangover_broccoli) (not (fears_hangover_satisfaction))))
 (:action succumb__hangover__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_hangover_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_hangover_tuna) (not (fears_hangover_satisfaction))))
 (:action succumb__hangover__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (fears_hangover_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_hangover_satisfaction)) (craves_hangover_turkey)))
 (:action succumb__hangover__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (fears_hangover_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_hangover_shrimp) (not (fears_hangover_satisfaction))))
 (:action succumb__hangover__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_satisfaction_sweetroll) (fears_hangover_satisfaction) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_hangover_sweetroll) (not (fears_hangover_satisfaction))))
 (:action overcome__jealousy__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_jealousy_tuna) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_jealousy_lubricity) (not (craves_jealousy_tuna))))
 (:action overcome__jealousy__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_scallop) (craves_jealousy_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_jealousy_intoxication) (not (craves_jealousy_scallop))))
 (:action succumb__dread__excitement__broccoli
  :parameters ()
  :precondition (and (fears_dread_excitement) (craves_excitement_broccoli) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (not (fears_dread_excitement)) (craves_dread_broccoli)))
 (:action overcome__anger__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_anger_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_anger_tuna)) (fears_anger_excitement)))
 (:action overcome__anger__excitement__sweetroll
  :parameters ()
  :precondition (and (craves_anger_sweetroll) (not (v2_4)) (craves_excitement_sweetroll) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anger_excitement) (not (craves_anger_sweetroll))))
 (:action overcome__anger__excitement__shrimp
  :parameters ()
  :precondition (and (craves_anger_shrimp) (craves_excitement_shrimp) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anger_excitement) (not (craves_anger_shrimp))))
 (:action overcome__anger__excitement__scallop
  :parameters ()
  :precondition (and (craves_anger_scallop) (not (v2_4)) (craves_excitement_scallop) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anger_excitement) (not (craves_anger_scallop))))
 (:action overcome__anger__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v2_4)) (craves_anger_broccoli) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anger_excitement) (not (craves_anger_broccoli))))
 (:action overcome__anger__excitement__chocolate
  :parameters ()
  :precondition (and (not (v2_4)) (craves_anger_chocolate) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_anger_excitement) (not (craves_anger_chocolate))))
 (:action overcome__anger__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (not (v2_4)) (craves_anger_turkey) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_anger_turkey)) (fears_anger_excitement)))
 (:action succumb__dread__satisfaction__chocolate
  :parameters ()
  :precondition (and (fears_dread_satisfaction) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_dread_chocolate) (not (fears_dread_satisfaction))))
 (:action succumb__dread__satisfaction__broccoli
  :parameters ()
  :precondition (and (fears_dread_satisfaction) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_dread_satisfaction)) (craves_dread_broccoli)))
 (:action succumb__dread__satisfaction__tuna
  :parameters ()
  :precondition (and (fears_dread_satisfaction) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_dread_satisfaction)) (craves_dread_tuna)))
 (:action succumb__dread__satisfaction__turkey
  :parameters ()
  :precondition (and (fears_dread_satisfaction) (craves_satisfaction_turkey) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_dread_turkey) (not (fears_dread_satisfaction))))
 (:action succumb__dread__satisfaction__shrimp
  :parameters ()
  :precondition (and (fears_dread_satisfaction) (craves_satisfaction_shrimp) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_dread_shrimp) (not (fears_dread_satisfaction))))
 (:action succumb__dread__satisfaction__sweetroll
  :parameters ()
  :precondition (and (fears_dread_satisfaction) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_dread_sweetroll) (not (fears_dread_satisfaction))))
 (:action succumb__dread__satisfaction__scallop
  :parameters ()
  :precondition (and (fears_dread_satisfaction) (craves_satisfaction_scallop) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (craves_dread_scallop) (not (fears_dread_satisfaction))))
 (:action succumb__dread__satisfaction__cherry
  :parameters ()
  :precondition (and (fears_dread_satisfaction) (craves_satisfaction_cherry) (not (of)))
  :effect (and (when (z_6_0) (v1_0)) (when (not (z_6_0)) (not (v1_0))) (when (z_6_1) (v1_1)) (when (not (z_6_1)) (not (v1_1))) (when (z_6_2) (v1_2)) (when (not (z_6_2)) (not (v1_2))) (when (z_6_3) (v1_3)) (when (not (z_6_3)) (not (v1_3))) (when (z_6_4) (v1_4)) (when (not (z_6_4)) (not (v1_4))) (when (and (not (v1_4)) (z_6_4)) (of)) (not (fears_dread_satisfaction)) (craves_dread_cherry)))
 (:action overcome__anger__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_satisfaction_turkey) (not (v1_4)) (craves_anger_turkey) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_anger_turkey)) (fears_anger_satisfaction)))
 (:action overcome__anger__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_anger_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_anger_tuna)) (fears_anger_satisfaction)))
 (:action overcome__anger__satisfaction__sweetroll
  :parameters ()
  :precondition (and (craves_anger_sweetroll) (not (v1_4)) (craves_satisfaction_sweetroll) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_anger_sweetroll)) (fears_anger_satisfaction)))
 (:action overcome__anger__satisfaction__shrimp
  :parameters ()
  :precondition (and (craves_satisfaction_shrimp) (craves_anger_shrimp) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_anger_shrimp)) (fears_anger_satisfaction)))
 (:action overcome__anger__satisfaction__scallop
  :parameters ()
  :precondition (and (craves_satisfaction_scallop) (craves_anger_scallop) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_anger_satisfaction) (not (craves_anger_scallop))))
 (:action overcome__anger__satisfaction__broccoli
  :parameters ()
  :precondition (and (not (v1_4)) (craves_anger_broccoli) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_anger_satisfaction) (not (craves_anger_broccoli))))
 (:action overcome__anger__satisfaction__chocolate
  :parameters ()
  :precondition (and (not (v1_4)) (craves_anger_chocolate) (craves_satisfaction_chocolate) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_anger_chocolate)) (fears_anger_satisfaction)))
 (:action succumb__hangover__lubricity__tuna
  :parameters ()
  :precondition (and (craves_lubricity_tuna) (fears_hangover_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_hangover_tuna) (not (fears_hangover_lubricity))))
 (:action overcome__anger__lubricity__shrimp
  :parameters ()
  :precondition (and (craves_anger_shrimp) (not (v11_4)) (craves_lubricity_shrimp) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anger_lubricity) (not (craves_anger_shrimp))))
 (:action overcome__anger__lubricity__scallop
  :parameters ()
  :precondition (and (not (v11_4)) (craves_anger_scallop) (craves_lubricity_scallop) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anger_lubricity) (not (craves_anger_scallop))))
 (:action overcome__anger__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_anger_broccoli) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anger_lubricity) (not (craves_anger_broccoli))))
 (:action overcome__anger__lubricity__chocolate
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_chocolate) (craves_anger_chocolate) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anger_lubricity) (not (craves_anger_chocolate))))
 (:action overcome__anger__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_turkey) (craves_anger_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anger_lubricity) (not (craves_anger_turkey))))
 (:action overcome__anger__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_anger_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anger_lubricity) (not (craves_anger_tuna))))
 (:action overcome__anger__lubricity__sweetroll
  :parameters ()
  :precondition (and (craves_anger_sweetroll) (not (v11_4)) (craves_lubricity_sweetroll) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_anger_lubricity) (not (craves_anger_sweetroll))))
 (:action succumb__hangover__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_hangover_intoxication)) (craves_hangover_chocolate)))
 (:action succumb__hangover__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_intoxication_broccoli) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_hangover_broccoli) (not (fears_hangover_intoxication))))
 (:action succumb__hangover__intoxication__tuna
  :parameters ()
  :precondition (and (fears_hangover_intoxication) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_hangover_intoxication)) (craves_hangover_tuna)))
 (:action succumb__hangover__intoxication__turkey
  :parameters ()
  :precondition (and (craves_intoxication_turkey) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_hangover_intoxication)) (craves_hangover_turkey)))
 (:action succumb__hangover__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_intoxication_shrimp) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_hangover_intoxication)) (craves_hangover_shrimp)))
 (:action succumb__hangover__intoxication__sweetroll
  :parameters ()
  :precondition (and (craves_intoxication_sweetroll) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_hangover_sweetroll) (not (fears_hangover_intoxication))))
 (:action succumb__hangover__intoxication__scallop
  :parameters ()
  :precondition (and (craves_intoxication_scallop) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_hangover_intoxication)) (craves_hangover_scallop)))
 (:action succumb__hangover__intoxication__cherry
  :parameters ()
  :precondition (and (craves_intoxication_cherry) (fears_hangover_intoxication) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_hangover_cherry) (not (fears_hangover_intoxication))))
 (:action overcome__anger__intoxication__sweetroll
  :parameters ()
  :precondition (and (craves_anger_sweetroll) (not (v8_4)) (craves_intoxication_sweetroll) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_anger_intoxication) (not (craves_anger_sweetroll))))
 (:action overcome__anger__intoxication__shrimp
  :parameters ()
  :precondition (and (craves_anger_shrimp) (not (v8_4)) (craves_intoxication_shrimp) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anger_shrimp)) (fears_anger_intoxication)))
 (:action overcome__anger__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_anger_scallop) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_anger_intoxication) (not (craves_anger_scallop))))
 (:action succumb__hangover__excitement__scallop
  :parameters ()
  :precondition (and (craves_excitement_scallop) (fears_hangover_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_hangover_scallop) (not (fears_hangover_excitement))))
 (:action overcome__anger__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_broccoli) (craves_anger_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_anger_intoxication) (not (craves_anger_broccoli))))
 (:action overcome__anger__intoxication__chocolate
  :parameters ()
  :precondition (and (craves_intoxication_chocolate) (not (v8_4)) (craves_anger_chocolate) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_anger_intoxication) (not (craves_anger_chocolate))))
 (:action overcome__anger__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_turkey) (craves_anger_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anger_turkey)) (fears_anger_intoxication)))
 (:action overcome__anger__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_anger_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_anger_tuna)) (fears_anger_intoxication)))
 (:action succumb__hangover__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (fears_hangover_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_hangover_broccoli) (not (fears_hangover_excitement))))
 (:action succumb__hangover__excitement__turkey
  :parameters ()
  :precondition (and (craves_excitement_turkey) (fears_hangover_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_hangover_turkey) (not (fears_hangover_excitement))))
 (:action succumb__hangover__excitement__chocolate
  :parameters ()
  :precondition (and (fears_hangover_excitement) (craves_excitement_chocolate) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_hangover_chocolate) (not (fears_hangover_excitement))))
 (:action succumb__hangover__excitement__sweetroll
  :parameters ()
  :precondition (and (craves_excitement_sweetroll) (fears_hangover_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_hangover_sweetroll) (not (fears_hangover_excitement))))
 (:action succumb__hangover__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (fears_hangover_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_hangover_tuna) (not (fears_hangover_excitement))))
 (:action succumb__hangover__excitement__cherry
  :parameters ()
  :precondition (and (craves_excitement_cherry) (fears_hangover_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_hangover_cherry) (not (fears_hangover_excitement))))
 (:action succumb__hangover__excitement__shrimp
  :parameters ()
  :precondition (and (craves_excitement_shrimp) (fears_hangover_excitement) (not (of)))
  :effect (and (when (z_8_0) (v2_0)) (when (not (z_8_0)) (not (v2_0))) (when (z_8_1) (v2_1)) (when (not (z_8_1)) (not (v2_1))) (when (z_8_2) (v2_2)) (when (not (z_8_2)) (not (v2_2))) (when (z_8_3) (v2_3)) (when (not (z_8_3)) (not (v2_3))) (when (z_8_4) (v2_4)) (when (not (z_8_4)) (not (v2_4))) (when (and (not (v2_4)) (z_8_4)) (of)) (craves_hangover_shrimp) (not (fears_hangover_excitement))))
 (:action overcome__dread__excitement__turkey
  :parameters ()
  :precondition (and (craves_dread_turkey) (craves_excitement_turkey) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (fears_dread_excitement) (not (craves_dread_turkey))))
 (:action succumb__boils__lubricity__tuna
  :parameters ()
  :precondition (and (fears_boils_lubricity) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_boils_tuna) (not (fears_boils_lubricity))))
 (:action overcome__dread__excitement__broccoli
  :parameters ()
  :precondition (and (craves_dread_broccoli) (craves_excitement_broccoli) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_dread_broccoli)) (fears_dread_excitement)))
 (:action succumb__boils__intoxication__scallop
  :parameters ()
  :precondition (and (fears_boils_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (not (fears_boils_intoxication)) (craves_boils_scallop)))
 (:action overcome__dread__satisfaction__broccoli
  :parameters ()
  :precondition (and (craves_dread_broccoli) (not (v1_4)) (craves_satisfaction_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_dread_broccoli)) (fears_dread_satisfaction)))
 (:action overcome__dread__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_dread_turkey) (craves_satisfaction_turkey) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_dread_turkey)) (fears_dread_satisfaction)))
 (:action overcome__dread__lubricity__broccoli
  :parameters ()
  :precondition (and (craves_dread_broccoli) (not (v11_4)) (craves_lubricity_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_dread_broccoli)) (fears_dread_lubricity)))
 (:action overcome__dread__lubricity__turkey
  :parameters ()
  :precondition (and (not (v11_4)) (craves_dread_turkey) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_dread_turkey)) (fears_dread_lubricity)))
 (:action overcome__dread__intoxication__turkey
  :parameters ()
  :precondition (and (not (v8_4)) (craves_dread_turkey) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_dread_intoxication) (not (craves_dread_turkey))))
 (:action overcome__dread__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_scallop) (craves_dread_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_dread_intoxication) (not (craves_dread_scallop))))
 (:action overcome__dread__intoxication__broccoli
  :parameters ()
  :precondition (and (craves_dread_broccoli) (not (v8_4)) (craves_intoxication_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_dread_broccoli)) (fears_dread_intoxication)))
 (:action overcome__hangover__excitement__broccoli
  :parameters ()
  :precondition (and (craves_excitement_broccoli) (not (v2_4)) (craves_hangover_broccoli) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_hangover_broccoli)) (fears_hangover_excitement)))
 (:action overcome__hangover__excitement__turkey
  :parameters ()
  :precondition (and (craves_hangover_turkey) (craves_excitement_turkey) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_hangover_turkey)) (fears_hangover_excitement)))
 (:action succumb__anxiety__lubricity__tuna
  :parameters ()
  :precondition (and (craves_lubricity_tuna) (fears_anxiety_lubricity) (not (of)))
  :effect (and (when (z_22_0) (v11_0)) (when (not (z_22_0)) (not (v11_0))) (when (z_22_1) (v11_1)) (when (not (z_22_1)) (not (v11_1))) (when (z_22_2) (v11_2)) (when (not (z_22_2)) (not (v11_2))) (when (z_22_3) (v11_3)) (when (not (z_22_3)) (not (v11_3))) (when (z_22_4) (v11_4)) (when (not (z_22_4)) (not (v11_4))) (when (and (not (v11_4)) (z_22_4)) (of)) (craves_anxiety_tuna) (not (fears_anxiety_lubricity))))
 (:action succumb__anxiety__intoxication__scallop
  :parameters ()
  :precondition (and (fears_anxiety_intoxication) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_18_0) (v8_0)) (when (not (z_18_0)) (not (v8_0))) (when (z_18_1) (v8_1)) (when (not (z_18_1)) (not (v8_1))) (when (z_18_2) (v8_2)) (when (not (z_18_2)) (not (v8_2))) (when (z_18_3) (v8_3)) (when (not (z_18_3)) (not (v8_3))) (when (z_18_4) (v8_4)) (when (not (z_18_4)) (not (v8_4))) (when (and (not (v8_4)) (z_18_4)) (of)) (craves_anxiety_scallop) (not (fears_anxiety_intoxication))))
 (:action overcome__hangover__satisfaction__broccoli
  :parameters ()
  :precondition (and (not (v1_4)) (craves_satisfaction_broccoli) (craves_hangover_broccoli) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_hangover_broccoli)) (fears_hangover_satisfaction)))
 (:action overcome__hangover__satisfaction__turkey
  :parameters ()
  :precondition (and (craves_hangover_turkey) (craves_satisfaction_turkey) (not (v1_4)) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_hangover_turkey)) (fears_hangover_satisfaction)))
 (:action overcome__hangover__lubricity__turkey
  :parameters ()
  :precondition (and (craves_hangover_turkey) (not (v11_4)) (craves_lubricity_turkey) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_hangover_turkey)) (fears_hangover_lubricity)))
 (:action overcome__hangover__lubricity__broccoli
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_broccoli) (craves_hangover_broccoli) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_hangover_broccoli)) (fears_hangover_lubricity)))
 (:action overcome__hangover__intoxication__broccoli
  :parameters ()
  :precondition (and (not (v8_4)) (craves_intoxication_broccoli) (craves_hangover_broccoli) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_broccoli))))
 (:action overcome__hangover__intoxication__turkey
  :parameters ()
  :precondition (and (craves_hangover_turkey) (not (v8_4)) (craves_intoxication_turkey) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_hangover_turkey)) (fears_hangover_intoxication)))
 (:action overcome__hangover__intoxication__scallop
  :parameters ()
  :precondition (and (not (v8_4)) (craves_hangover_scallop) (craves_intoxication_scallop) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_scallop))))
 (:action overcome__grief2__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_grief2_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (fears_grief2_lubricity) (not (craves_grief2_tuna))))
 (:action overcome__grief2__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_grief2_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_grief2_tuna)) (fears_grief2_intoxication)))
 (:action overcome__loneliness__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_loneliness_tuna) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_loneliness_tuna)) (fears_loneliness_lubricity)))
 (:action overcome__loneliness__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_loneliness_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_loneliness_intoxication) (not (craves_loneliness_tuna))))
 (:action overcome__grief2__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_grief2_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_grief2_tuna)) (fears_grief2_excitement)))
 (:action overcome__grief2__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_grief2_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_grief2_tuna)) (fears_grief2_satisfaction)))
 (:action overcome__loneliness__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (craves_loneliness_tuna) (not (v2_4)) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_loneliness_tuna)) (fears_loneliness_excitement)))
 (:action overcome__loneliness__satisfaction__tuna
  :parameters ()
  :precondition (and (craves_loneliness_tuna) (not (v1_4)) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_loneliness_tuna)) (fears_loneliness_satisfaction)))
 (:action overcome__dread__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_dread_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_dread_tuna)) (fears_dread_excitement)))
 (:action overcome__dread__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_dread_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (not (craves_dread_tuna)) (fears_dread_satisfaction)))
 (:action overcome__dread__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_lubricity_tuna) (craves_dread_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_dread_tuna)) (fears_dread_lubricity)))
 (:action overcome__dread__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_dread_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (not (craves_dread_tuna)) (fears_dread_intoxication)))
 (:action overcome__hangover__excitement__tuna
  :parameters ()
  :precondition (and (craves_excitement_tuna) (not (v2_4)) (craves_hangover_tuna) (not (of)))
  :effect (and (when (z_1_0) (v2_0)) (when (not (z_1_0)) (not (v2_0))) (when (z_1_1) (v2_1)) (when (not (z_1_1)) (not (v2_1))) (when (z_1_2) (v2_2)) (when (not (z_1_2)) (not (v2_2))) (when (z_1_3) (v2_3)) (when (not (z_1_3)) (not (v2_3))) (when (z_1_4) (v2_4)) (when (not (z_1_4)) (not (v2_4))) (when (and (v2_4) (not (z_1_4))) (of)) (not (craves_hangover_tuna)) (fears_hangover_excitement)))
 (:action overcome__hangover__satisfaction__tuna
  :parameters ()
  :precondition (and (not (v1_4)) (craves_hangover_tuna) (craves_satisfaction_tuna) (not (of)))
  :effect (and (when (z_0_0) (v1_0)) (when (not (z_0_0)) (not (v1_0))) (when (z_0_1) (v1_1)) (when (not (z_0_1)) (not (v1_1))) (when (z_0_2) (v1_2)) (when (not (z_0_2)) (not (v1_2))) (when (z_0_3) (v1_3)) (when (not (z_0_3)) (not (v1_3))) (when (z_0_4) (v1_4)) (when (not (z_0_4)) (not (v1_4))) (when (and (v1_4) (not (z_0_4))) (of)) (fears_hangover_satisfaction) (not (craves_hangover_tuna))))
 (:action overcome__hangover__lubricity__tuna
  :parameters ()
  :precondition (and (not (v11_4)) (craves_hangover_tuna) (craves_lubricity_tuna) (not (of)))
  :effect (and (when (z_21_0) (v11_0)) (when (not (z_21_0)) (not (v11_0))) (when (z_21_1) (v11_1)) (when (not (z_21_1)) (not (v11_1))) (when (z_21_2) (v11_2)) (when (not (z_21_2)) (not (v11_2))) (when (z_21_3) (v11_3)) (when (not (z_21_3)) (not (v11_3))) (when (z_21_4) (v11_4)) (when (not (z_21_4)) (not (v11_4))) (when (and (v11_4) (not (z_21_4))) (of)) (not (craves_hangover_tuna)) (fears_hangover_lubricity)))
 (:action overcome__hangover__intoxication__tuna
  :parameters ()
  :precondition (and (not (v8_4)) (craves_hangover_tuna) (craves_intoxication_tuna) (not (of)))
  :effect (and (when (z_11_0) (v8_0)) (when (not (z_11_0)) (not (v8_0))) (when (z_11_1) (v8_1)) (when (not (z_11_1)) (not (v8_1))) (when (z_11_2) (v8_2)) (when (not (z_11_2)) (not (v8_2))) (when (z_11_3) (v8_3)) (when (not (z_11_3)) (not (v8_3))) (when (z_11_4) (v8_4)) (when (not (z_11_4)) (not (v8_4))) (when (and (v8_4) (not (z_11_4))) (of)) (fears_hangover_intoxication) (not (craves_hangover_tuna))))
)
