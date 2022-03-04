create table population
(
    guid        integer not null
        constraint population_pk
            primary key,
    name        varchar not null,
    description varchar
);

alter table population
    owner to bioexplorer;

create unique index population_guid_uindex
    on population (guid);

create unique index population_name_uindex
    on population (name);

create table model_template
(
    guid        integer not null
        constraint model_template_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table model_template
    owner to bioexplorer;

create unique index model_template_guid_uindex
    on model_template (guid);

create table node_type
(
    guid        integer not null
        constraint node_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table node_type
    owner to bioexplorer;

create unique index node_type_guid_uindex
    on node_type (guid);

create table model_type
(
    guid        integer not null
        constraint model_type_pk
            primary key,
    code        varchar not null,
    description integer
);

alter table model_type
    owner to bioexplorer;

create unique index model_type_guid_uindex
    on model_type (guid);

create table morphology_type
(
    guid        integer not null
        constraint morphology_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table morphology_type
    owner to bioexplorer;

create unique index morphology_type_guid_uindex
    on morphology_type (guid);

create table configuration
(
    guid  varchar not null
        constraint configuration_pk
            primary key,
    value varchar not null
);

alter table configuration
    owner to bioexplorer;

create unique index configuration_guid_uindex
    on configuration (guid);

create table section_type
(
    guid        integer not null
        constraint section_type_pk
            primary key,
    description varchar not null
);

alter table section_type
    owner to bioexplorer;

create unique index section_type_guid_uindex
    on section_type (guid);

create unique index section_type_description_uindex
    on section_type (description);

create table electrical_type
(
    guid        integer not null
        constraint electrical_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table electrical_type
    owner to bioexplorer;

create unique index electrical_type_guid_uindex
    on electrical_type (guid);

create unique index electrical_type_code_uindex
    on electrical_type (code);

create table morphological_type
(
    guid        integer not null
        constraint morphological_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table morphological_type
    owner to bioexplorer;

create unique index morphological_type_code_uindex
    on morphological_type (code);

create unique index morphological_type_guid_uindex
    on morphological_type (guid);

create table morphology
(
    guid     integer not null
        constraint morphology_pk
            primary key,
    basename varchar not null
);

alter table morphology
    owner to bioexplorer;

create unique index morphology_basename_uindex
    on morphology (basename);

create unique index morphology_guid_uindex
    on morphology (guid);

create table node
(
    guid             integer                    not null
        constraint node_pk
            primary key,
    population_guid  integer          default 0 not null,
    x                double precision           not null,
    y                double precision           not null,
    z                double precision           not null,
    rotation_angle_x double precision default 0 not null,
    rotation_angle_y double precision default 0 not null,
    rotation_angle_z double precision default 0 not null,
    morphology_guid  integer          default 0 not null
        constraint node_morphology_guid_fk
            references morphology
            on update cascade on delete cascade
        constraint node_morphology_guid_fk_2
            references morphology
            on update cascade on delete cascade,
    e_type_guid      integer          default 0 not null
        constraint node_electrical_type_guid_fk
            references electrical_type
            on update cascade on delete cascade,
    m_type_guid      integer          default 0 not null
);

alter table node
    owner to bioexplorer;

create index node_population_guid_index
    on node (population_guid);

create table section
(
    morphology_guid     integer not null,
    section_guid        integer not null,
    section_parent_guid integer not null,
    section_type_guid   integer not null
        constraint section_section_type_guid_fk
            references section_type,
    points              bytea   not null,
    constraint section_pk
        primary key (morphology_guid, section_guid, section_parent_guid)
);

alter table section
    owner to bioexplorer;

create unique index section_morphology_guid_section_guid_uindex
    on section (morphology_guid, section_guid);

create index section_morphology_guid_index
    on section (morphology_guid);

