;;
;; Test problem for PDDL extension features (arrays + bounded integers).
;;
;; Two players start with empty boards and zero best scores.
;; The goal is to have:
;;   - player1's board set to [0, 2, 3]
;;   - player2's board set to [1, 0, 4]
;;   - both players' best scores updated to the highest slot value
;;

(define (problem test-features-p1)
    (:domain test-features)

    (:objects
        player1 player2 - player
    )

    (:init
        ;; Array initial values via array.mk
        (= (slots player1) (array.mk (0 0 0)))
        (= (slots player2) (array.mk (0 0 0)))

        ;; Bounded integer initial values
        (= (best player1) 0)
        (= (best player2) 0)
    )

    (:goal
        (and
            ;; Array goal: check individual slots via read
            (= (read (slots player1) (1)) 2)
            (= (read (slots player1) (2)) 3)

            ;; Array goal: compare whole array to a constant
            (= (slots player2) (array.mk (1 0 4)))

            ;; Bounded integer goal
            (= (best player1) 3)
            (= (best player2) 4)
        )
    )

)
