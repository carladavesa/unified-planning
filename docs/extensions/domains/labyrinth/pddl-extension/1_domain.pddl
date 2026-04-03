(define (domain labyrinth)
    (:requirements :typing :arrays :bounded-integers)

    (:types
        direction   - object
        idx         - (number 0 3)
        pos         - (number 0 15)
        dir_set     - (set direction)
        grid        - (array 4 4 dir_set)
    )

    (:constants
        N S E W - direction
    )

    (:functions
        (card_at)   - grid
        (robot_at)  - pos
        ;;(robot_r) - idx
        ;;(robot_c) - idx
    )

    (:action move_north
        :parameters (?r ?c - idx)
        :precondition (and
            (= (robot_at) (+ (* ?r 4) ?c))
            (member N (read (card_at) ?r ?c))
            (member S (read (card_at) (- ?r 1) ?c))
        )
        :effect (and
            (write (robot_at) (+ (* (- ?r 1) 4) ?c))
        )
    )

    (:action move_south
        :parameters (?r ?c - idx)
        :precondition (and
            (= (robot_at) (+ (* ?r 4) ?c))
            (member N (read (card_at) ?r ?c))
            (member S (read (card_at) (+ ?r 1) ?c))
        )
        :effect (and
            (write (robot_at) (+ (* (+ ?r 1) 4) ?c))
        )
    )

    (:action move_east
        :parameters (?r ?c - idx)
        :precondition (and
            (= (robot_at) (+ (* ?r 4) ?c))
            (member E (read (card_at) ?r ?c))
            (member W (read (card_at) (+ ?r 1) ?c))
        )
        :effect (and
            (write (robot_at) (+ (* (+ ?r 1) 4) ?c))
        )
    )

    (:action move_west
        :parameters (?r ?c - idx)
        :precondition (and
            (= (robot_at) (+ (* ?r 4) ?c))
            (member W (read (card_at) ?r ?c))
            (member E (read (card_at) (- ?r 1) ?c))
        )
        :effect (and
            (write (robot_at) (+ (* (- ?r 1) 4) ?c))
        )
    )

    (:action rotate_col_up
        :parameters (?c - idx)
        :precondition (and
            (forall (?r - idx)
                (not
                    (= (robot_at) (+ (* ?r 4) ?c))
                )
            )
        )
        :effect (and
            (forall (?r - idx)
                (when (>= ?r 1)
                    (write ((card_at) (- ?r 1) ?c) (read (card_at) ?r ?c))
                )
            )
            (write ((card_at) 3 ?c) (read (card_at) 0 ?c))
        )
    )

    (:action rotate_col_down
        :parameters (?c - idx)
        :precondition (and
            (forall (?r - idx)
                (not
                    (= (robot_at) (+ (* ?r 4) ?c))
                )
            )
        )
        :effect (and
            (forall (?r - idx)
                (when (>= ?r 1)
                    (write ((card_at) ?r ?c) (read (card_at) (- ?r 1) ?c))
                )
            )
            (write ((card_at) 0 ?c) (read (card_at) 3 ?c))
        )
    )

    (:action rotate_row_left
        :parameters (?r - idx)
        :precondition (and
            (forall (?c - idx)
                (not
                    (= (robot_at) (+ (* ?r 4) ?c))
                )
            )
        )
        :effect (and
            (forall (?c - idx)
                (when (>= ?c 1)
                    (write ((card_at) ?r (- ?c 1)) (read (card_at) ?r ?c))
                )
            )
            (write ((card_at) ?r 3) (read (card_at) ?r 0))
        )
    )

    (:action rotate_row_right
        :parameters (?r - idx)
        :precondition (and
            (forall (?c - idx)
                (not
                    (= (robot_at) (+ (* ?r 4) ?c))
                )
            )
        )
        :effect (and
            (forall (?c - idx)
                (when (>= ?c 1)
                    (write ((card_at) ?r ?c) (read (card_at) ?r (- ?c 1)))
                )
            )
            (write ((card_at) ?r 0) (read (card_at) ?r 3))
        )
    )
)
